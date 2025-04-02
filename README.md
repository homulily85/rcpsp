[English](#english)|[Tiếng Việt](#tiếng-việt)
## Notes
- test.py is used for testing purposes.
- benchmark.py is used to benchmark the performance of encoders.
- For staircase and thesis encoders, you need to run on Linux (as the pypblib library doesn't work on Windows).

## Command Line Arguments

## English
### Notes
- test.py is used for testing purposes.
- For staircase and thesis encoders, you need to run on Linux (as the pypblib library doesn't work on Windows).
### Command Line Arguments
```
python benchmark.py dataset_name encoder_type timeout [options]
```

**Required Arguments:**
- `dataset_name`: Name of the dataset to benchmark
- `encoder_type`: Type of encoder to use (choices: 'thesis', 'staircase', 'lia')
- `timeout`: Timeout for solving in seconds (use 0 for no timeout)

**Optional Arguments:**
- `--show_solution`: Save the best solution to a file
- `--verify`: Verify the result after solving
- `--verbose`: Display logs in terminal during execution

## Tiếng Việt
## Ghi chú
- test.py được sử dụng cho mục đích test.
- benchmark.py được sử dụng để đánh giá hiệu suất của các phương pháp mã hóa.
- Đối với mã hóa staircase và thesis, bạn cần chạy trên Linux (vì thư viện pypblib không hoạt động trên Windows).
### Tham số dòng lệnh
```
python benchmark.py tên_bộ_dữ_liệu loại_mã_hóa thời_gian_chờ [tùy_chọn]
```

**Tham số bắt buộc:**
- `tên_bộ_dữ_liệu`: Tên của bộ dữ liệu cần đánh giá
- `loại_mã_hóa`: Loại mã hóa sử dụng (lựa chọn: 'thesis', 'staircase', 'lia')
- `thời_gian_chờ`: Thời gian chờ tối đa cho quá trình giải (tính bằng giây, sử dụng 0 để không giới hạn thời gian)

**Tham số tùy chọn:**
- `--show_solution`: Lưu lời giải tốt nhất vào file
- `--verify`: Kiểm tra tính đúng đắn của lời giải sau khi giải
- `--verbose`: Hiển thị log trong terminal trong quá trình thực thi