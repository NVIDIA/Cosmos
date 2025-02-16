import os
import sys
from datetime import datetime
import base64
from io import BytesIO
import pytest

import torch
from torchvision.utils import make_grid
from torchvision.transforms import CenterCrop

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from huggingface_hub import hf_hub_download

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(project_root)
from cosmos1.models.tokenizer.inference.video_lib import CausalVideoTokenizer
from cosmos1.models.tokenizer.inference.utils import read_video

# global config
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# test configs
TEST_CONFIGS = [
    ("CV4x8x8", 'nvidia/Cosmos-0.1-Tokenizer-CV4x8x8'),
    ("CV8x8x8", 'nvidia/Cosmos-0.1-Tokenizer-CV8x8x8'),
    ("CV8x16x16", 'nvidia/Cosmos-0.1-Tokenizer-CV8x16x16'),
    ("DV4x8x8", 'nvidia/Cosmos-0.1-Tokenizer-DV4x8x8'),
    ("DV8x8x8", 'nvidia/Cosmos-0.1-Tokenizer-DV8x8x8'),
    ("DV8x16x16", 'nvidia/Cosmos-0.1-Tokenizer-DV8x16x16'),
    ("CV8x8x8", 'nvidia/Cosmos-1.0-Tokenizer-CV8x8x8'),
    ("DV8x16x16", 'nvidia/Cosmos-1.0-Tokenizer-DV8x16x16'),
]


# HTML Logger as a pytest fixture
@pytest.fixture(scope="module")
def html_logger():
    class HTMLLogger:
        def __init__(self):
            self.log_entries = []
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def log(self, message: str):
            self.log_entries.append(f'<div class="log-entry">{message}</div>')
            print(f'{message}')

        def log_comparison(self, fig, model_id: str):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()

            html = f'''
            <div class="comparison">
                <h3>Quality Check Visualization - {model_id}</h3>
                <img src="data:image/png;base64,{img_str}" style="max-width:100%;">
            </div>
            '''
            self.log_entries.append(html)

        def save(self):
            html_content = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tokenizer Test Results - {self.timestamp}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background: #f5f5f5;
                    }}
                    .log-entry {{
                        padding: 5px 10px;
                        margin: 5px 0;
                        border-radius: 4px;
                        background: #fff;
                    }}
                    .comparison {{
                        margin: 20px 0;
                        padding: 20px;
                        background: #fff;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    h3 {{
                        color: #333;
                        margin-top: 0;
                    }}
                    img {{
                        display: block;
                        margin: 10px auto;
                    }}
                </style>
            </head>
            <body>
                <h1>Tokenizer Test Results</h1>
                <p>Generated on: {self.timestamp}</p>
                {''.join(self.log_entries)}
            </body>
            </html>
            '''

            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f'tokenizer_test_{self.timestamp}.html')

            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"\nLog saved to {log_file_path}\n")

    logger = HTMLLogger()
    yield logger
    logger.save()


@pytest.fixture(scope="module")
def video_tensor():
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data", "video.mp4")
    video = read_video(video_path)

    assert video.shape[0] >= 17, "Video length should be at least 17 frames"
    assert video.shape[1] >= 512, "Video height should be at least 512 pixels"
    assert video.shape[2] >= 512, "Video width should be at least 512 pixels"
    assert video.shape[3] == 3, "Video should have 3 channels"

    input_tensor = CenterCrop(512)(
        torch.from_numpy(
            video[np.newaxis, ...]
        )[:, :17].to('cuda').to(torch.bfloat16).permute(0, 4, 1, 2, 3) / 255.0 * 2.0 - 1.0
    )
    return input_tensor


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_tokenizer(config, html_logger, video_tensor):
    name, model_id = config
    continuous = name.startswith(("C", 'c'))
    temporal_compression, spatial_compression = list(map(int, name[2:].split('x')))[:2]
    print(f"\nTesting tokenizer: {model_id}")

    # Load models
    checkpoint_enc = hf_hub_download(
        repo_id=model_id,
        repo_type="model",
        filename="encoder.jit",
    )
    checkpoint_dec = hf_hub_download(
        repo_id=model_id,
        repo_type="model",
        filename="decoder.jit",
    )

    encoder = CausalVideoTokenizer(checkpoint_enc=checkpoint_enc)
    decoder = CausalVideoTokenizer(checkpoint_dec=checkpoint_dec)

    try:
        # Test shape check
        reconstructed_tensor = auto_shape_check(
            video_tensor, encoder, decoder,
            temporal_compression, spatial_compression,
            continuous, model_id, html_logger
        )

        # Test quality check
        manual_quality_check(video_tensor, reconstructed_tensor, model_id, html_logger)

        html_logger.log(f"✓ {model_id} - All tests passed")

    finally:
        # Cleanup
        del encoder
        del decoder
        del reconstructed_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def auto_shape_check(input_tensor, encoder, decoder, temporal_compression,
                     spatial_compression, continuous, model_id, html_logger):
    if continuous:
        (latent,) = encoder.encode(input_tensor)
        torch.testing.assert_close(latent.shape, (
            1, 16, (17 - 1) // temporal_compression + 1,
            512 // spatial_compression, 512 // spatial_compression))
        reconstructed_tensor = decoder.decode(latent)
    else:
        (indices, codes) = encoder.encode(input_tensor)
        torch.testing.assert_close(indices.shape, (
            1, (17 - 1) // temporal_compression + 1,
            512 // spatial_compression, 512 // spatial_compression))
        torch.testing.assert_close(codes.shape, (
            1, 6, (17 - 1) // temporal_compression + 1,
            512 // spatial_compression, 512 // spatial_compression))
        reconstructed_tensor = decoder.decode(indices)

    torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
    html_logger.log(f"✓ {model_id} - Auto shape check passed")
    return reconstructed_tensor


def manual_quality_check(x, x_rec, model_id, html_logger):
    check_result = [False]

    x = (x[0, :, -1, ...].cpu().float().clamp(-1, 1) + 1.0) / 2.0
    x_rec = (x_rec[0, :, -1, ...].cpu().float().clamp(-1, 1) + 1.0) / 2.0

    x_grid = make_grid(x, nrow=4, normalize=False)
    x_rec_grid = make_grid(x_rec, nrow=4, normalize=False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f'Model: {model_id}', fontsize=10)

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(x_grid.permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed')
    plt.imshow(x_rec_grid.permute(1, 2, 0))
    plt.axis('off')

    plt.subplots_adjust(bottom=0.2)

    ax_yes = plt.axes([0.3, 0.05, 0.2, 0.075])
    ax_no = plt.axes([0.55, 0.05, 0.2, 0.075])

    btn_yes = Button(ax_yes, 'Accept', color='lightgreen')
    btn_no = Button(ax_no, 'Reject', color='lightcoral')

    def on_click_yes(event):
        check_result[0] = True
        plt.close(fig)

    def on_click_no(event):
        check_result[0] = False
        plt.close(fig)

    btn_yes.on_clicked(on_click_yes)
    btn_no.on_clicked(on_click_no)

    html_logger.log_comparison(fig, model_id)
    plt.show()

    assert check_result[0], f'✗ {model_id} - Human quality check failed'
    html_logger.log(f'✓ {model_id} - Human quality check passed')

    if __name__ == '__main__':
        pytest.main([__file__, '-v'])