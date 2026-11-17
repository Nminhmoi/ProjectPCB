### Kiểm nhánh dự phòng SIFT (`--min-ncc 0.999 --fallback SIFT`)

<sub>Sinh bởi `experiments/check_sift_fallback.py` trên 3 cặp; max_features = 2000; mặc định pipeline: `DEFAULT_MIN_NCC = 0.90`, không dự phòng.</sub>

| Cặp | NCC ORB | NCC SIFT | ORB (ms) | SIFT (ms) | H ORB vs SIFT (px) | Lệch sẵn (px) | GT ORB | GT SIFT | Ép SIFT → dùng | Thời gian mặc định → ép (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_missing_hole_01` | 0.9984 | 0.9984 | 191 | 1378 | 0.00 | 0.0 | 3/3 | 3/3 | ORB * | 225 → 1495 |
| `06_spur_08` | 0.9960 | 0.9959 | 292 | 1934 | 0.31 | 46.8 | 5/5 | 5/5 | ORB * | 256 → 2041 |
| `04_spurious_copper_05` | 0.9999 | 0.9999 | 247 | 2081 | 0.01 | 0.0 | 0/3 | 0/3 | ORB | 256 → 238 |

<sub>\* không phương án nào đạt ngưỡng ép; `align_images_robust` dùng H có NCC cao nhất.</sub>

**Kết quả:**
- Ép ngưỡng 0.999: nhánh SIFT được chạy ở 2/3 cặp (kết quả cuối dùng SIFT ở 0 cặp; 2 cặp không phương án nào đạt ngưỡng), thời gian tăng trung bình **+1012 ms/cặp** (tối đa +1785 ms).
- H của ORB và SIFT lệch nhau tối đa **0.31 px** tại 4 góc ảnh; số lỗi GT phát hiện được **giống nhau ở 3/3 cặp**. NCC của ORB thấp nhất 0.9960 — cao hơn nhiều ngưỡng mặc định 0.90.
- Trên 693 cặp (`experiments/summary.csv`): SIFT căn chính xác hơn (corner trung vị 0.035 so với 0.280 px) nhưng chậm 6.4 lần (1485 so với 233 ms) và F1 chỉ hơn +0.0029 vì bước trừ vi sai đã chừa dung sai ±2 px.

**Kết luận — khi nào bật `--fallback SIFT`:**
- **Không bật mặc định**: trên PKU, ORB luôn cho NCC ≥ 0.995 (CODE_REVIEW.md §2.1), nhánh dự phòng không bao giờ chạy với ngưỡng 0.90; ép chạy thì tốn thêm thời gian mà kết quả phát hiện không đổi.
- **Nên bật** khi ORB thực sự hỏng: căn chỉnh thất bại hình học, hoặc NCC < 0.90 — ảnh camera thật ít chi tiết (vùng phủ sơn lớn), mờ, lệch sáng mạnh, hoặc biến dạng lớn. Khi đó SIFT chỉ chạy cho số ít cặp bị từ chối nên chi phí trung bình nhỏ.
- **Chế độ độ chính xác cao (offline)**: dùng thẳng `--detector SIFT` thay vì dự phòng, nếu cần sai số căn chỉnh < 0.1 px và chấp nhận ~1.5 s/cặp.
