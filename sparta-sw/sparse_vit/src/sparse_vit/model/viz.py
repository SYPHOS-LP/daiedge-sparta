import numpy as np
import matplotlib.pyplot as plt


def plot_vit_positional_embedding_similarities(
    pe,
    grid_size=(14, 14),
    reference_patches=None,
    cmap="viridis",
    figsize=(10, 10),
):
    """
    Plot cosine similarity maps between selected positional embeddings
    and all other patch positional embeddings.

    Parameters
    ----------
    pe : np.ndarray
        Positional embeddings of shape (n, d_model), e.g. (196, d_model).
    grid_size : tuple[int, int]
        Patch grid size, e.g. (14, 14).
    reference_patches : list[int] | None
        Patch indices to use as similarity anchors.
        If None, uses a few representative patches.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Figure size.
    """
    pe = np.asarray(pe)

    h, w = grid_size
    n, _ = pe.shape

    if n != h * w:
        raise ValueError(
            f"Expected n = grid_size[0] * grid_size[1] = {h*w}, "
            f"but got pe shape {pe.shape}."
        )

    # Normalize embeddings for cosine similarity
    pos_norm = pe / np.linalg.norm(pe, axis=1, keepdims=True).clip(min=1e-8)

    if reference_patches is None:
        reference_patches = [
            0,  # top-left
            w // 2,  # top-center
            w - 1,  # top-right
            (h // 2) * w,  # center-left
            (h // 2) * w + w // 2,  # center
            (h // 2) * w + w - 1,  # center-right
            (h - 1) * w,  # bottom-left
            (h - 1) * w + w // 2,  # bottom-center
            h * w - 1,  # bottom-right
        ]

    num_refs = len(reference_patches)
    cols = int(np.ceil(np.sqrt(num_refs)))
    rows = int(np.ceil(num_refs / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, ref_idx in zip(axes, reference_patches):
        if ref_idx < 0 or ref_idx >= n:
            raise ValueError(
                f"Reference patch index {ref_idx} is out of bounds for n={n}."
            )

        similarities = pos_norm @ pos_norm[ref_idx]
        sim_map = similarities.reshape(h, w)

        im = ax.imshow(sim_map, cmap=cmap, vmin=-1, vmax=1)
        ref_y, ref_x = divmod(ref_idx, w)

        ax.scatter(ref_x, ref_y, c="red", s=40, marker="x")
        ax.set_title(f"Patch {ref_idx} ({ref_y}, {ref_x})")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[num_refs:]:
        ax.axis("off")

    fig.colorbar(im, ax=axes[:num_refs], shrink=0.7, label="Cosine similarity")
    fig.suptitle("ViT positional embedding similarity per patch", fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


def plot_pes(pe: np.ndarray):
    """
    Plot positional embeddings
    """

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pe, cmap="viridis")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.colorbar(label="Embedding Value")
    plt.title("Sparse Haar-like Positional Embedding Heatmap")
    plt.show()
