# 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性： 
# 
#  
#  每行中的整数从左到右按升序排列。 
#  每行的第一个整数大于前一行的最后一个整数。 
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
# 输出：true
#  
# 
#  示例 2： 
# 
#  
# 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
# 输出：false
#  
# 
#  
# 
#  提示： 
# 
#  
#  m == matrix.length 
#  n == matrix[i].length 
#  1 <= m, n <= 100 
#  -104 <= matrix[i][j], target <= 104 
#  
#  Related Topics 数组 二分查找 
#  👍 313 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 特例
        if not matrix:
            return False
        # 线性扫描
        index = -1
        for i in range(len(matrix)):
            if target <= matrix[i][-1]:
                index = i
                break
        if index == -1:
            return False  # target不在矩阵里
        else:  # 二分查找
            left, right = 0, len(matrix[index]) - 1
            while left <= right:
                mid = (left + right) // 2
                if matrix[index][mid] == target:
                    return True
                elif matrix[index][mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return False

# leetcode submit region end(Prohibit modification and deletion)
