- test.py sử dụng với mục đích chạy thử.
- benchmark.py sử dụng để đánh giá các phương pháp mã hóa đối với từng bộ dữ liệu cụ thể. File này cần được chạy với tham số dòng lệnh với cú pháp: python benchmark.py [tên bộ dữ liệu] [tên phương pháp mã hóa (lia, staircase, thesis)] [time out (giây)] [--verify (để cho phép kiểm tra - tùy chọn)]
  Đối với staircase và thesis cần phải chạy trên Linux (do hạn thư viện pypblib không hoạt động trên Ưindows)
