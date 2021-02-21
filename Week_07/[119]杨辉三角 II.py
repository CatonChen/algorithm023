# 给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。 
# 
#  
# 
#  在杨辉三角中，每个数是它左上方和右上方的数的和。 
# 
#  示例: 
# 
#  输入: 3
# 输出: [1,3,3,1]
#  
# 
#  进阶： 
# 
#  你可以优化你的算法到 O(k) 空间复杂度吗？ 
#  Related Topics 数组 
#  👍 251 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [1] * (rowIndex + 1)
        for i in range(1, len(res) - 1):
            # print(res[i - 1])
            # print((rowIndex - i + 1) // i)
            res[i] = res[i - 1] * (rowIndex - i + 1) // i
        return res
# leetcode submit region end(Prohibit modification and deletion)
