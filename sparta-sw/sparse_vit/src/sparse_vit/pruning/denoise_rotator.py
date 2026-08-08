import types
from typing import Optional

import torch
from torch import nn, matmul


def vit_fuse_rms_single_layer(layer):
    layer.self_attn.q_proj.weight.data = (
        layer.self_attn.q_proj.weight.data @ torch.diag(layer.norm1.weight.data)
    )
    layer.self_attn.k_proj.weight.data = (
        layer.self_attn.k_proj.weight.data @ torch.diag(layer.norm1.weight.data)
    )
    layer.self_attn.v_proj.weight.data = (
        layer.self_attn.v_proj.weight.data @ torch.diag(layer.norm1.weight.data)
    )
    layer.norm1.weight.data = torch.ones_like(
        layer.norm1.weight.data,
        dtype=layer.norm1.weight.dtype,
        device=layer.norm1.weight.device,
    )

    layer.mlp.fc1.weight.data = layer.mlp.fc1.weight.data @ torch.diag(
        layer.norm2.weight.data
    )
    layer.norm2.weight.data = torch.ones_like(
        layer.norm2.weight.data,
        dtype=layer.norm2.weight.dtype,
        device=layer.norm2.weight.device,
    )


@torch.compile
def row_entropy_sum(matrix):
    """
    Estimate row entropy sums.
    """
    abs_sq = torch.nan_to_num(matrix, nan=0.0, posinf=1e5, neginf=0)
    row_sums = torch.sum(abs_sq, dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)

    probs = abs_sq / row_sums
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1e5, neginf=0)
    probs = torch.where(probs > 0, probs, 1)
    log_probs = torch.log(probs)
    log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=1e5, neginf=0)

    entropies = -torch.sum(probs * log_probs, dim=1)
    res = torch.sum(entropies)

    if torch.isnan(res).any():
        print("WARNING")

    res = torch.nan_to_num(res, nan=0.0, posinf=0, neginf=0)

    return res


class RotatorOptimizer(nn.Module):
    def __init__(
        self,
        weight_dict_list,
        r_dim,
        num_attention_heads,
        head_dim,
        device,
        positive=True,
        hessian_dict_list=None,
        num_piece=1,
    ):
        """ """
        super().__init__()

        self.weight_dict_list = weight_dict_list
        self.num_piece = num_piece
        self.r_dim = r_dim
        self.device = device
        self.A_dim = self.r_dim // self.num_piece
        self.A_list = [
            nn.Parameter(torch.eye(self.A_dim, device=device))
            for _ in range(self.num_piece)
        ]
        self.positive = positive
        self.hessian_dict_list = hessian_dict_list
        self.num_layer = len(weight_dict_list)
        self.B_list_list = []
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.dtype = torch.bfloat16

        for _ in range(self.num_layer):
            self.B_list_list.append(
                [
                    nn.Parameter(torch.eye(self.head_dim, device=device))
                    for _ in range(self.num_attention_heads)
                ]
            )

        for idx in range(self.num_layer):
            for name in self.weight_dict_list[idx]:
                self.weight_dict_list[idx][name] = (
                    self.weight_dict_list[idx][name]
                    .weight.detach()
                    .to(self.device)
                    .to(self.dtype)
                )
                self.weight_dict_list[idx][name].requires_grad_(False)

            for name in self.hessian_dict_list[idx]:
                self.hessian_dict_list[idx][name] = (
                    self.hessian_dict_list[idx][name]
                    .detach()
                    .to(self.device)
                    .to(self.dtype)
                )
                self.hessian_dict_list[idx][name].requires_grad_(False)

    def parameters(self, recurse=True):
        res = []
        for l in self.A_list:
            res.append(l)
        for l in self.B_list_list:
            res += l
        return res

    def get_orthogonal_matrix(self):
        Q = torch.block_diag(
            *[torch.linalg.qr(self.A_list[i])[0] for i in range(self.num_piece)]
        ).to(dtype=self.dtype)
        return Q

    def get_orthogonal_matrix_R2_list_list(self):
        R2_list_list = []
        for i in range(self.num_layer):
            R2_list = []
            for j in range(self.num_attention_heads):
                R2_list.append(
                    torch.linalg.qr(self.B_list_list[i][j])[0].to(dtype=self.dtype)
                )
            R2_list_list.append(R2_list)
        return R2_list_list

    def get_R1_list(self):
        return [torch.linalg.qr(self.A_list[i])[0] for i in range(self.num_piece)]

    def get_R2_list_list(self):
        return self.get_orthogonal_matrix_R2_list_list()

    def compute_salience_RWX(self, weight, hessian, R):
        raise NotImplementedError

    def compute_salience_WR_1RX(self, weight, hessian, R):
        raise NotImplementedError

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        raise NotImplementedError

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        raise NotImplementedError

    def forward(self, indices_dict=None):
        """ """
        R = self.get_orthogonal_matrix()
        R2_list_list = self.get_orthogonal_matrix_R2_list_list()
        loss = None

        WR_1RX_list = ["self_attn.q_proj", "self_attn.k_proj", "mlp.fc1"]
        RWX_list = ["mlp.fc2"]
        R2WR_1RX_list = ["self_attn.v_proj"]
        RWR2_1R2X_list = ["self_attn.o_proj"]

        hidden_list = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"]
        intermediate_list = ["mlp.fc1", "mlp.fc2"]

        for idx in range(self.num_layer):
            for name in RWX_list:
                weight = self.weight_dict_list[idx][name]
                hessian = self.hessian_dict_list[idx][name]

                if indices_dict is not None:
                    if name in hidden_list:
                        weight = weight[:, indices_dict["hidden"]]
                        hessian = hessian[indices_dict["hidden"], :][
                            :, indices_dict["hidden"]
                        ]
                    elif name in intermediate_list:
                        weight = weight[:, indices_dict["intermediate"]]
                        hessian = hessian[indices_dict["intermediate"], :][
                            :, indices_dict["intermediate"]
                        ]

                salience = self.compute_salience_RWX(
                    weight,
                    hessian,
                    R,
                )
                layer_loss = row_entropy_sum(salience.T)
                loss = layer_loss if loss is None else loss + layer_loss

            for name in WR_1RX_list:
                weight = self.weight_dict_list[idx][name]

                if indices_dict is not None:
                    if name in hidden_list:
                        weight = weight[indices_dict["hidden"], :]
                    elif name in intermediate_list:
                        weight = weight[indices_dict["intermediate"], :]

                hessian = self.hessian_dict_list[idx][name]
                salience = self.compute_salience_WR_1RX(weight, hessian, R)
                layer_loss = row_entropy_sum(salience)
                loss = layer_loss if loss is None else loss + layer_loss

            for name in R2WR_1RX_list:

                weight = self.weight_dict_list[idx][name]
                hessian = self.hessian_dict_list[idx][name]
                salience = self.compute_salience_R2WR_1RX(
                    weight, hessian, R, R2_list_list[idx]
                )
                layer_loss = row_entropy_sum(salience) + row_entropy_sum(salience.T)
                loss = layer_loss if loss is None else loss + layer_loss

            for name in RWR2_1R2X_list:

                weight = self.weight_dict_list[idx][name]
                hessian = self.hessian_dict_list[idx][name]
                salience = self.compute_salience_RWR2_1R2X(
                    weight, hessian, R, R2_list_list[idx]
                )
                layer_loss = row_entropy_sum(salience) + row_entropy_sum(salience.T)
                loss = layer_loss if loss is None else loss + layer_loss

        return loss if self.positive else -loss


@torch.compile
def compute_salience_RWX_wanda(weight, hessian, R):
    x_norms = torch.abs(torch.diag(hessian))
    return ((R.T @ weight) ** 2) * x_norms


@torch.compile
def compute_salience_WR_1RX_wanda(weight, hessian, R):
    rotated_hessian = R.T @ hessian @ R
    x_norms = torch.abs(torch.diag(rotated_hessian))
    return ((weight @ R) ** 2) * x_norms


@torch.compile
def compute_salience_R2WR_1RX_wanda(weight, hessian, R, R2_list):
    R2 = torch.block_diag(*R2_list)
    rotated_hessian = R.T @ hessian @ R
    x_norms = torch.abs(torch.diag(rotated_hessian))
    return ((R2.T @ weight @ R) ** 2) * x_norms


@torch.compile
def compute_salience_RWR2_1R2X_wanda(weight, hessian, R, R2_list):
    r2_list = []
    for r2 in R2_list:
        for _ in range(weight.shape[1] // r2.shape[0] // len(R2_list)):
            r2_list.append(r2)
    R2 = torch.block_diag(*r2_list)
    rotated_hessian = R2.T @ hessian @ R2
    x_norms = torch.abs(torch.diag(rotated_hessian))
    return ((R.T @ weight @ R2) ** 2) * x_norms


class RotatorOptimizer_wanda(RotatorOptimizer):
    def compute_salience_RWX(self, weight, hessian, R):
        return compute_salience_RWX_wanda(weight, hessian, R)

    def compute_salience_WR_1RX(self, weight, hessian, R):
        return compute_salience_WR_1RX_wanda(weight, hessian, R)

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        return compute_salience_R2WR_1RX_wanda(weight, hessian, R, R2_list)

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        return compute_salience_RWR2_1R2X_wanda(weight, hessian, R, R2_list)


class RotatorOptimizer_magnitude(RotatorOptimizer):
    def compute_salience_RWX(self, weight, hessian, R):
        return (R.T @ weight) ** 2

    def compute_salience_WR_1RX(self, weight, hessian, R):
        return (weight @ R) ** 2

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        R2 = torch.block_diag(*R2_list)
        return (R2.T @ weight @ R) ** 2

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        r2_list = []
        for r2 in R2_list:
            for _ in range(weight.shape[1] // r2.shape[0] // len(R2_list)):
                r2_list.append(r2)
        R2 = torch.block_diag(*r2_list)
        return (R.T @ weight @ R2) ** 2


@torch.compile
def compute_salience_RWX_sparsegpt(weight, hessian, R):
    hinv_diag = torch.abs(torch.diag(hessian))
    return ((R.T @ weight) ** 2) / hinv_diag


@torch.compile
def compute_salience_WR_1RX_sparsegpt(weight, hessian, R):
    rotated_hinv = R.T @ hessian @ R
    hinv_diag = torch.abs(torch.diag(rotated_hinv))
    return ((weight @ R) ** 2) / hinv_diag


@torch.compile
def compute_salience_R2WR_1RX_sparsegpt(weight, hessian, R, R2_list):
    R2 = torch.block_diag(*R2_list)
    rotated_hinv = R.T @ hessian @ R
    hinv_diag = torch.abs(torch.diag(rotated_hinv))
    return ((R2.T @ weight @ R) ** 2) / hinv_diag


@torch.compile
def compute_salience_RWR2_1R2X_sparsegpt(weight, hessian, R, R2_list):
    r2_list = []
    for r2 in R2_list:
        for _ in range(weight.shape[1] // r2.shape[0] // len(R2_list)):
            r2_list.append(r2)
    R2 = torch.block_diag(*r2_list)
    rotated_hinv = R2.T @ hessian @ R2
    hinv_diag = torch.abs(torch.diag(rotated_hinv))
    return ((R.T @ weight @ R2) ** 2) / hinv_diag


class RotatorOptimizer_sparsegpt(RotatorOptimizer):
    def __init__(
        self,
        weight_dict_list,
        r_dim,
        num_attention_heads,
        head_dim,
        device,
        positive=True,
        hessian_dict_list=None,
        num_piece=1,
        percdamp=0.01,
    ):
        super().__init__(
            weight_dict_list,
            r_dim,
            num_attention_heads,
            head_dim,
            device,
            positive=positive,
            hessian_dict_list=hessian_dict_list,
            num_piece=num_piece,
        )
        self.inverse_hessian(percdamp=percdamp)

    def inverse_hessian(self, percdamp=0.01):
        hinv_dict_list = []
        for idx in range(self.num_layer):
            hinv_dict_list.append({})
            for name in self.hessian_dict_list[idx]:
                H = self.hessian_dict_list[idx][name].to(dtype=torch.float32)
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(H.shape[0], device=H.device)
                H[diag, diag] += damp

                success = False
                attempts = 0
                while not success:
                    try:
                        H = torch.inverse(H)
                        success = True
                    except RuntimeError:
                        print(
                            f"Attempt {attempts}: Matrix not positive "
                            "definite, modifying diagonal elements."
                        )
                    H[diag, diag] += damp
                    attempts += 1

                hinv_dict_list[idx][name] = H.to(dtype=torch.bfloat16)

        self.hessian_dict_list = hinv_dict_list
        torch.cuda.empty_cache()

    def compute_salience_RWX(self, weight, hessian, R):
        return compute_salience_RWX_sparsegpt(weight, hessian, R)

    def compute_salience_WR_1RX(self, weight, hessian, R):
        return compute_salience_WR_1RX_sparsegpt(weight, hessian, R)

    def compute_salience_R2WR_1RX(self, weight, hessian, R, R2_list):
        return compute_salience_R2WR_1RX_sparsegpt(weight, hessian, R, R2_list)

    def compute_salience_RWR2_1R2X(self, weight, hessian, R, R2_list):
        return compute_salience_RWR2_1R2X_sparsegpt(weight, hessian, R, R2_list)


def vit_fuse_rotation_single_layer(layer, R1, R2_list):
    R2_list_o = []
    for r2 in R2_list:
        for _ in range(
            layer.self_attn.v_proj.weight.data.shape[1]
            // layer.self_attn.v_proj.weight.data.shape[0]
        ):
            R2_list_o.append(r2)

    R2_transform_o = torch.block_diag(*R2_list_o).to(R1.device)
    R2_transform_v = torch.block_diag(*R2_list).to(R1.device)

    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data @ R1.T
    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data @ R1.T
    layer.self_attn.v_proj.weight.data = (
        R2_transform_v.T @ layer.self_attn.v_proj.weight.data @ R1.T
    )
    layer.self_attn.o_proj.weight.data = (
        R1 @ layer.self_attn.o_proj.weight.data @ R2_transform_o
    )
    layer.mlp.fc1.weight.data = layer.mlp.fc1.weight.data @ R1.T
    layer.mlp.fc2.weight.data = R1 @ layer.mlp.fc2.weight.data


def rotated_vit_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    **kwargs,
):
    if self.R1 is not None:
        hidden_states = matmul(hidden_states, self.R1.T)

    residual = hidden_states
    hidden_states = self.norm1(hidden_states)

    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    if self.R1 is not None:
        hidden_states = matmul(hidden_states, self.R1)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    return outputs


def replace_vit_layer(model, layer_idx, R1):
    layer = model.blocks[layer_idx]
    dtype = next(layer.parameters()).dtype
    device = next(layer.parameters()).device

    layer.R1 = nn.Parameter(R1.to(dtype=dtype, device=device))
    layer.forward = types.MethodType(rotated_vit_block_forward, layer)
    model.blocks[layer_idx] = layer


def load_rotated_vit(model, state_dict_path):
    hidden_size = model.embed_dim
    for idx in range(len(model.blocks)):
        R1 = torch.eye(hidden_size)
        replace_vit_layer(model, idx, R1)

    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model
