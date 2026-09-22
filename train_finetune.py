"""
KNPS 2026 주제3(불법 시설물 변화 탐지) — 합성 데이터 + AI Hub 실데이터 혼합 파인튜닝 스크립트
=============================================================================
이 대회는 학습/평가 위성영상을 전혀 제공하지 않습니다(예시 영상 한 장도 없음). 그래서 이 스크립트는
두 종류의 학습 신호를 합칩니다.

  1) 합성(synthetic) 데이터 — 절차적으로 생성한 pre/post 쌍으로 "변화가 아닌 차이"
     (계절색 차이, 그림자 방향, 촬영조건별 밝기·색조 차이, 수 화소 시차)에 반응하지 않도록
     하드 네거티브를 대량으로 노출시킵니다. 완전 무데이터 상황에서도 만들 수 있습니다.
  2) AI Hub 실데이터 — 주최측이 학습 활용을 공식 허용한 AI Hub "AI기반 국립공원 변화탐지
     모니터링 플랫폼 구축"(dataSetSn=71363, Skyset 플랫폼) 데이터셋에서 실제 위성영상 pre/post
     쌍과 변화 마스크를 뽑아 만든 패치 zip(`aihub_real_pairs.zip`)을 사용합니다. 진짜 위성영상의
     노이즈·질감 분포를 학습에 반영하기 위함입니다.

`--real-data-ratio`로 두 데이터 소스의 혼합 비율을 조절합니다 (0 = 합성 데이터만 사용 = 원래 스크립트와
동일 동작). AI Hub zip이 없으면(`--aihub-zip` 미지정) 자동으로 합성 데이터만 사용합니다.

⚠️ 실험 결과 요약(README.md 참고): 실데이터 혼합만으로는 Public 리더보드 점수 개선이 거의 없었음
(0.2827 → 0.2830). 반면 추론 단계의 `CONF_THRESH`를 0.8 → 0.5로 낮춘 것은 훨씬 큰 개선을 만들어냄
(0.2830 → 0.2931). 즉 이 스크립트로 체크포인트를 다시 학습하는 것보다, predict.ipynb의 후처리
파라미터를 조정하는 쪽이 더 저비용·고효율일 가능성이 높습니다.

Colab에서 실행하는 법
---------------------
1) 베이스라인 패키지를 세션에 올리고 압축 해제 (predict.ipynb, assets/, requirements.txt 등)
2) !pip install -q segmentation-models-pytorch shapely
3) (선택) AI Hub 패치 zip을 준비해 로컬 디스크로 복사 (Drive 마운트 상태로 직접 읽으면 매우 느립니다)
   !cp "/content/drive/MyDrive/aihub_real_pairs.zip" /content/aihub_real_pairs.zip
4) 이 스크립트를 실행:
   !python train_finetune.py \\
     --ckpt-in "assets/model/unet_r18_cd.pt" \\
     --ckpt-out "assets/model/unet_r18_cd_ft2.pt" \\
     --aihub-zip "/content/aihub_real_pairs.zip" \\
     --steps 3000 --batch-size 16 --real-data-ratio 0.5
   원본 체크포인트(unet_r18_cd.pt)는 절대 덮어쓰지 않고 별도 파일로 저장됩니다. 반드시 Public
   리더보드 점수로 원본 대비 개선 여부를 확인한 뒤 채택하세요 (하루 제출 횟수 제한 유의).
"""

import argparse
import csv
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

H, W = 256, 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------------------
# 1. 절차적(procedural) 배경 텍스처 생성 — 실제 위성영상이 없으므로 완전 합성
# ------------------------------------------------------------------------------
def perlin_like_noise(h: int, w: int, octaves: int = 4) -> np.ndarray:
    """여러 스케일의 업샘플된 랜덤 노이즈를 겹쳐 자연스러운 지형 질감을 흉내냅니다."""
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        scale = 2 ** o
        small = np.random.randn(max(2, h // (4 * scale)), max(2, w // (4 * scale))).astype(np.float32)
        small_t = torch.from_numpy(small)[None, None]
        big = F.interpolate(small_t, size=(h, w), mode="bicubic", align_corners=False)[0, 0].numpy()
        out += amp * big
        total += amp
        amp *= 0.5
    out /= total
    out -= out.min()
    out /= out.max() + 1e-6
    return out


def make_background(h: int, w: int, kind: str) -> np.ndarray:
    """kind: 'forest' | 'bareland' | 'field' | 'mixed' — 대략적인 색조만 흉내냄"""
    base = perlin_like_noise(h, w, octaves=5)
    fine = perlin_like_noise(h, w, octaves=2) * 0.3
    tex = np.clip(base * 0.7 + fine, 0, 1)
    if kind == "forest":
        color = np.array([40, 90, 40], np.float32)
        var = np.array([35, 45, 30], np.float32)
    elif kind == "bareland":
        color = np.array([150, 120, 90], np.float32)
        var = np.array([40, 35, 30], np.float32)
    elif kind == "field":
        color = np.array([120, 140, 70], np.float32)
        var = np.array([40, 40, 30], np.float32)
    else:
        color = np.array([110, 110, 100], np.float32)
        var = np.array([50, 50, 50], np.float32)
    img = color[None, None, :] + (tex[..., None] - 0.5) * 2 * var[None, None, :]
    img += np.random.randn(h, w, 3).astype(np.float32) * 4.0  # 센서 노이즈
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------
# 2. 광도(photometric) 하드 네거티브 변환 — "변화가 아닌 차이"를 흉내냄
# ------------------------------------------------------------------------------
def photometric_jitter(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """밝기/대비/감마/색조를 독립적으로 흔들어 계절·촬영조건 차이를 흉내냅니다."""
    x = img.astype(np.float32) / 255.0
    gamma = np.exp(np.random.uniform(-0.35, 0.35) * strength)
    x = np.clip(x, 1e-4, 1) ** gamma
    contrast = 1.0 + np.random.uniform(-0.25, 0.25) * strength
    x = np.clip((x - 0.5) * contrast + 0.5, 0, 1)
    brightness = np.random.uniform(-0.12, 0.12) * strength
    x = np.clip(x + brightness, 0, 1)
    hue_shift = (np.random.uniform(-0.08, 0.08, size=3) * strength).astype(np.float32)
    x = np.clip(x + hue_shift[None, None, :], 0, 1)
    return (x * 255).astype(np.uint8)


def add_shadow_band(img: np.ndarray) -> np.ndarray:
    """그림자 방향/길이 차이를 흉내내는 대각선 명암 그라디언트."""
    h, w = img.shape[:2]
    angle = np.random.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (xx * np.cos(angle) + yy * np.sin(angle)) / (h + w)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
    darkness = 1.0 - 0.35 * np.random.uniform(0.2, 1.0) * (grad[..., None] ** 2)
    return np.clip(img.astype(np.float32) * darkness, 0, 255).astype(np.uint8)


def small_parallax_warp(img: np.ndarray, max_shift_px: float = 3.0) -> np.ndarray:
    """시차로 인한 수 화소 어긋남을 흉내내는 미소 아핀 이동/회전."""
    h, w = img.shape[:2]
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    dx = np.random.uniform(-max_shift_px, max_shift_px) / (w / 2)
    dy = np.random.uniform(-max_shift_px, max_shift_px) / (h / 2)
    theta = np.random.uniform(-0.01, 0.01)
    mat = torch.tensor([[np.cos(theta), -np.sin(theta), dx],
                         [np.sin(theta), np.cos(theta), dy]], dtype=torch.float32)[None]
    grid = F.affine_grid(mat, t.shape, align_corners=False)
    warped = F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return warped[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()


def moving_vehicle_blob(img: np.ndarray) -> np.ndarray:
    """움직이는 차량 + 주변 색번짐을 흉내내는 작은 타원 얼룩(무변화 라벨 유지용)."""
    out = img.copy()
    h, w = out.shape[:2]
    cy, cx = np.random.randint(20, h - 20), np.random.randint(20, w - 20)
    ry, rx = np.random.randint(2, 4), np.random.randint(4, 8)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    color = np.random.randint(30, 220, size=3)
    out[mask] = (0.6 * color + 0.4 * out[mask]).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------
# 3. 합성 포지티브 — 신축(사각 지붕) / 벌목(질감 블롭 제거)
# ------------------------------------------------------------------------------
def add_synthetic_building(post: np.ndarray, mask: np.ndarray, label: int) -> None:
    h, w = post.shape[:2]
    bw, bh = np.random.randint(14, 40), np.random.randint(14, 40)
    cx, cy = np.random.randint(bw, w - bw), np.random.randint(bh, h - bh)
    angle = np.random.uniform(0, 90)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.deg2rad(angle)
    rx = (xx - cx) * np.cos(rad) + (yy - cy) * np.sin(rad)
    ry = -(xx - cx) * np.sin(rad) + (yy - cy) * np.cos(rad)
    roof_mask = (np.abs(rx) <= bw / 2) & (np.abs(ry) <= bh / 2)
    roof_palette = [np.array([200, 200, 205]), np.array([170, 60, 50]), np.array([90, 100, 110]), np.array([230, 230, 220])]
    roof_color = roof_palette[np.random.randint(len(roof_palette))]
    grad = (rx / (bw + 1e-6) * 20).astype(np.float32)
    post[roof_mask] = np.clip(roof_color[None, :] + grad[roof_mask, None], 0, 255).astype(np.uint8)
    mask[roof_mask] = label


def remove_synthetic_trees(pre: np.ndarray, post: np.ndarray, mask: np.ndarray, label: int) -> None:
    h, w = post.shape[:2]
    cy, cx = np.random.randint(20, h - 20), np.random.randint(20, w - 20)
    r = np.random.randint(15, 45)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    # 전(pre) 영상에는 수관 텍스처를 덧그려 "원래 숲이었다"를 표현
    canopy = make_background(h, w, "forest")
    pre[blob] = canopy[blob]
    # 후(post) 영상은 맨땅/풀밭으로 교체
    bare = make_background(h, w, random.choice(["bareland", "field"]))
    post[blob] = bare[blob]
    mask[blob] = label


# ------------------------------------------------------------------------------
# 4. 합성 데이터셋
# ------------------------------------------------------------------------------
class SyntheticChangeDataset(IterableDataset):
    """무한 스트림. pre, post, mask(0=배경,1=new_building,2=tree_removal)를 냅니다."""

    def __init__(self, hard_negative_ratio: float = 0.6, seed: int | None = None):
        self.hard_negative_ratio = hard_negative_ratio
        self.rng = random.Random(seed)

    def _one(self):
        bg_kind = random.choice(["forest", "bareland", "field", "mixed"])
        pre = make_background(H, W, bg_kind)
        post = pre.copy()
        mask = np.zeros((H, W), np.uint8)

        # 항상: 두 시점 간 광도 차이 + 시차를 독립적으로 부여 (실제 대회 데이터의 핵심 함정)
        post = photometric_jitter(post, strength=1.0)
        pre = photometric_jitter(pre, strength=0.4)
        if random.random() < 0.7:
            post = add_shadow_band(post)
        post = small_parallax_warp(post, max_shift_px=3.0)
        if random.random() < 0.3:
            post = moving_vehicle_blob(post)
        if random.random() < 0.3:
            pre = moving_vehicle_blob(pre)

        is_hard_negative = random.random() < self.hard_negative_ratio
        if not is_hard_negative:
            if random.random() < 0.5:
                add_synthetic_building(post, mask, label=1)
            else:
                remove_synthetic_trees(pre, post, mask, label=2)

        return pre, post, mask

    def __iter__(self):
        while True:
            yield self._one()


# ------------------------------------------------------------------------------
# 5. AI Hub 실데이터셋 — Skyset 플랫폼 pre/post 패치 + 변화 마스크
# ------------------------------------------------------------------------------
def load_aihub_manifest(zf: zipfile.ZipFile):
    """zip 안의 manifest.csv(id, has_building, has_treeremoval 열)를 읽어
    건물신축/벌목/무변화 카테고리별 id 목록으로 나눕니다."""
    with zf.open("manifest.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
        building_ids, treeremoval_ids, negative_ids = [], [], []
        for row in reader:
            pid = row["id"]
            has_b = row.get("has_building", "0") in ("1", "True", "true")
            has_t = row.get("has_treeremoval", "0") in ("1", "True", "true")
            if has_b:
                building_ids.append(pid)
            elif has_t:
                treeremoval_ids.append(pid)
            else:
                negative_ids.append(pid)
    return building_ids, treeremoval_ids, negative_ids


class AihubRealDataset(IterableDataset):
    """AI Hub Skyset 변화탐지 패치 zip에서 실제 위성영상 pre/post/mask를 읽는 무한 스트림.

    zip 구조: images/<id>/{pre.png,post.png,mask_building.png,mask_treeremoval.png} + manifest.csv
    (AI Hub 원본 LABEL 라스터를 4x4로 타일링해 256x256 패치로 만들고, 정답 마스크를 미리 계산해 둔 것.
    자세한 변환 파이프라인은 README.md의 "AI Hub 실데이터 확보 및 가공" 절 참고)
    """

    def __init__(self, zip_path: str, building_prob: float = 0.4, treeremoval_prob: float = 0.4,
                 seed: int | None = None):
        self.zip_path = zip_path
        self.building_prob = building_prob
        self.treeremoval_prob = treeremoval_prob
        self.rng = random.Random(seed)
        self._zf = zipfile.ZipFile(zip_path)
        self.building_ids, self.treeremoval_ids, self.negative_ids = load_aihub_manifest(self._zf)
        print(f"[AihubRealDataset] building={len(self.building_ids)} "
              f"treeremoval={len(self.treeremoval_ids)} negative={len(self.negative_ids)}")

    def _read_png(self, name: str) -> np.ndarray:
        with self._zf.open(name) as f:
            with Image.open(io.BytesIO(f.read())) as im:
                return np.asarray(im.copy())

    def _one(self):
        r = self.rng.random()
        if r < self.building_prob and self.building_ids:
            pid = self.rng.choice(self.building_ids)
        elif r < self.building_prob + self.treeremoval_prob and self.treeremoval_ids:
            pid = self.rng.choice(self.treeremoval_ids)
        else:
            pid = self.rng.choice(self.negative_ids)

        pre = self._read_png(f"images/{pid}/pre.png")[..., :3]
        post = self._read_png(f"images/{pid}/post.png")[..., :3]
        mb = self._read_png(f"images/{pid}/mask_building.png")
        mt = self._read_png(f"images/{pid}/mask_treeremoval.png")

        mask = np.zeros((H, W), np.uint8)
        mask[mt.astype(bool)] = 2
        mask[mb.astype(bool)] = 1  # 겹치면 건물 우선

        # 좌우반전 + 90도 회전 증강만 적용 (픽셀값 자체는 건드리지 않아 실제 노이즈 특성 보존)
        k = self.rng.randint(0, 3)
        pre, post, mask = np.rot90(pre, k), np.rot90(post, k), np.rot90(mask, k)
        if self.rng.random() < 0.5:
            pre, post, mask = np.fliplr(pre), np.fliplr(post), np.fliplr(mask)

        return np.ascontiguousarray(pre), np.ascontiguousarray(post), np.ascontiguousarray(mask)

    def __iter__(self):
        while True:
            yield self._one()


class MixedChangeDataset(IterableDataset):
    """합성 데이터와 AI Hub 실데이터를 real_ratio 확률로 섞는 스트림.
    real_ds가 None이면 real_ratio를 강제로 0으로 만들어 순수 합성 데이터로 동작합니다
    (--aihub-zip 미지정 시 기존 스크립트와 동일한 동작을 보장)."""

    def __init__(self, synthetic_ds: SyntheticChangeDataset, real_ds: "AihubRealDataset | None",
                 real_ratio: float = 0.5, seed: int | None = None):
        self.synthetic_ds = synthetic_ds
        self.real_ds = real_ds
        self.real_ratio = real_ratio if real_ds is not None else 0.0
        self.rng = random.Random(seed)

    def __iter__(self):
        syn_it = iter(self.synthetic_ds)
        real_it = iter(self.real_ds) if self.real_ds is not None else None
        while True:
            if real_it is not None and self.rng.random() < self.real_ratio:
                yield next(real_it)
            else:
                yield next(syn_it)


def collate(batch):
    pres, posts, masks = zip(*batch)
    x = np.stack([
        np.concatenate([(p.astype(np.float32) / 255 - MEAN) / STD,
                         (q.astype(np.float32) / 255 - MEAN) / STD], axis=2)
        for p, q in zip(pres, posts)
    ]).astype(np.float32)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    y = torch.from_numpy(np.stack(masks)).long()
    return x, y


# ------------------------------------------------------------------------------
# 6. 학습 루프
# ------------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice[1:].mean()  # 배경 제외, 변화 클래스 위주로 최적화


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-in", default="assets/model/unet_r18_cd.pt")
    ap.add_argument("--ckpt-out", default="assets/model/unet_r18_cd_ft.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze-encoder", action="store_true", default=True)
    ap.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    ap.add_argument("--hard-negative-ratio", type=float, default=0.6)
    ap.add_argument("--aihub-zip", default=None,
                     help="AI Hub 실데이터 패치 zip 경로 (images/<id>/{pre,post,mask_*}.png + manifest.csv)")
    ap.add_argument("--real-data-ratio", type=float, default=0.5,
                     help="--aihub-zip 지정 시, 각 스텝에서 실데이터를 뽑을 확률")
    ap.add_argument("--real-building-prob", type=float, default=0.4)
    ap.add_argument("--real-treeremoval-prob", type=float, default=0.4)
    ap.add_argument("--save-every", type=int, default=300,
                     help="Colab 런타임 disconnect에 대비해 N 스텝마다 ckpt-out에 중간 저장")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=6, classes=3)
    ckpt = torch.load(args.ckpt_in, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print("인코더 동결: 디코더/세그멘테이션 헤드만 미세조정합니다.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    synthetic_ds = SyntheticChangeDataset(hard_negative_ratio=args.hard_negative_ratio, seed=args.seed)
    real_ds = None
    if args.aihub_zip:
        real_ds = AihubRealDataset(args.aihub_zip, building_prob=args.real_building_prob,
                                    treeremoval_prob=args.real_treeremoval_prob, seed=args.seed)
    ds = MixedChangeDataset(synthetic_ds, real_ds, real_ratio=args.real_data_ratio, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0)

    # 클래스 불균형 대응: 배경 비중이 압도적이므로 변화 클래스에 가중치
    class_weight = torch.tensor([1.0, 4.0, 4.0], device=device)

    model.train()
    it = iter(dl)
    running = 0.0
    for step in range(1, args.steps + 1):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=class_weight) + 0.5 * dice_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if step % 100 == 0:
            print(f"step {step}/{args.steps}  loss={running / 100:.4f}")
            running = 0.0
        if args.save_every and step % args.save_every == 0:
            Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict()}, args.ckpt_out)
            print(f"(중간 저장 완료: {args.ckpt_out}, step {step})")

    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.ckpt_out)
    print(f"학습 완료, 최종 저장: {args.ckpt_out}")
    print("반드시 Public 리더보드 제출로 원본/이전 체크포인트 대비 점수를 비교한 뒤 채택 여부를 정하세요.")


if __name__ == "__main__":
    main()
