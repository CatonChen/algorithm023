# 给定一个非空二维矩阵 matrix 和一个整数 k，找到这个矩阵内部不大于 k 的最大矩形和。 
# 
#  示例: 
# 
#  输入: matrix = [[1,0,1],[0,-2,3]], k = 2
# 输出: 2 
# 解释: 矩形区域 [[0, 1], [-2, 3]] 的数值和是 2，且 2 是不超过 k 的最大数字（k = 2）。
#  
# 
#  说明： 
# 
#  
#  矩阵内的矩形区域面积必须大于 0。 
#  如果行数远大于列数，你将如何解答呢？ 
#  
#  Related Topics 队列 二分查找 动态规划 
#  👍 152 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        import bisect  # 二分查找模块
        row = len(matrix)  # 行
        col = len(matrix[0])  # 列
        res = float('-inf')
        for left in range(col):
            # left边界下的每行总和
            _sum = [0] * row
            # print(_sum)
            for right in range(left, col):
                # print(right)
                for j in range(row):
                    _sum[j] += matrix[j][right]
                    # print(_sum)
                arr = [0]  # 中间数组
                cur = 0
                for tmp in _sum:
                    cur += tmp  # 前缀和
                    # 二分查找，找出最接近cur-k的下标
                    loc = bisect.bisect_left(arr, cur - k)
                    if loc < len(arr):
                        res = max(cur - arr[loc], res)  # 将最接近k的前缀和赋予res
                    bisect.insort(arr, cur)  # 将前缀和放入res并排序
                    # print(arr)
        return res

# leetcode submit region end(Prohibit modification and deletion)
