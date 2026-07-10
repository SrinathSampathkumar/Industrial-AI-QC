"""
GradCAM / EigenCAM-style heatmap for YOLO detection.

Strategy: Register a forward hook on the last backbone feature layer
(before the FPN neck and Detect head). Run the full YOLO inference
(through the proper ultralytics wrapper) so that all FPN Concat /
Upsample wiring is handled correctly.  Extract activations from the
hook and reduce them to a spatial heatmap.
"""

import cv2
import numpy as np
import torch


class YOLOGradCAM:
    """
    EigenCAM-style activation heatmap for any YOLOv8/v11 model.

    The hook is attached to the last 'C2f', 'C3k2', 'C3', or 'SPPF'
    layer found in the backbone — whichever appears last in the model
    tree.  If none of those are found we fall back to the second-to-last
    direct child of the model.Sequential.
    """

    _BACKBONE_TYPES = {"C2f", "C3k2", "C3", "SPPF", "C2fAttn",
                       "C3k", "C2", "Bottleneck", "BottleneckCSP"}

    def __init__(self, yolo_model):
        """
        Parameters
        ----------
        yolo_model : ultralytics.YOLO
            A loaded ultralytics YOLO model instance.
        """
        self.yolo = yolo_model               # keep the full wrapper for inference
        self.inner_model = yolo_model.model  # DetectionModel (nn.Module)
        self._activations = None
        self._hook_handle = None
        self._attach_hook()

    # ── Hook management ───────────────────────────────────────

    def _attach_hook(self):
        target = self._find_target_layer()
        if target is None:
            return
        self._hook_handle = target.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        """Store the last seen feature map."""
        if isinstance(output, torch.Tensor):
            self._activations = output.detach()
        elif isinstance(output, (list, tuple)):
            # Take the last tensor in the output list
            for item in reversed(output):
                if isinstance(item, torch.Tensor):
                    self._activations = item.detach()
                    break

    def _find_target_layer(self):
        """
        Walk the model.model Sequential and return the last module whose
        class name matches a known backbone block.
        """
        seq = self.inner_model.model          # nn.Sequential of YOLO layers
        best = None
        for layer in seq:
            if type(layer).__name__ in self._BACKBONE_TYPES:
                best = layer
        if best is not None:
            return best
        # Fallback: second-to-last child
        children = list(seq.children())
        if len(children) >= 2:
            return children[-2]
        return None

    # ── Public API ────────────────────────────────────────────

    def generate(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """
        Generate a normalised CAM heatmap for *img_bgr*.

        Parameters
        ----------
        img_bgr : np.ndarray
            Input image in BGR format (HxWx3 uint8).

        Returns
        -------
        cam : np.ndarray | None
            Float32 array in [0, 1] with the same spatial size as the
            input image, or None if the hook produced no activations.
        """
        if self._hook_handle is None:
            return None

        self._activations = None

        # Run YOLO inference — this goes through the full model pipeline
        # including preprocessing, FPN, and Detect.  The hook fires on
        # the registered backbone layer regardless.
        with torch.no_grad():
            self.yolo.predict(img_bgr, verbose=False, save=False)

        if self._activations is None:
            return None

        act = self._activations                # shape: (1, C, H, W) typically
        if act.dim() < 3:
            return None
        if act.dim() == 3:
            act = act.unsqueeze(0)             # ensure (B, C, H, W)

        # Mean over channel dimension → (1, H, W) → (H, W)
        cam = act.mean(dim=1).squeeze().cpu().float().numpy()
        cam = np.maximum(cam, 0)

        if cam.ndim == 0:                      # scalar edge case
            return None
        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = img_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam_resized

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()


# ── Utility ───────────────────────────────────────────────────

def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Blend a JET colourmap of *cam* onto *img_bgr*.

    Parameters
    ----------
    img_bgr : np.ndarray  — original BGR image
    cam     : np.ndarray  — float32 heatmap in [0, 1], same spatial size
    alpha   : float       — weight of original image (1-alpha for heatmap)

    Returns
    -------
    overlay : np.ndarray  — blended BGR image
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, alpha, heatmap, 1 - alpha, 0)
    return overlay
