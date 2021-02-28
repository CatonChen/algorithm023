# 给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。 
# 
#  示例 1: 
# 
#  输入: 2
# 输出: [0,1,1] 
# 
#  示例 2: 
# 
#  输入: 5
# 输出: [0,1,1,2,1,2] 
# 
#  进阶: 
# 
#  
#  给出时间复杂度为O(n*sizeof(integer))的解答非常容易。但你可以在线性时间O(n)内用一趟扫描做到吗？ 
#  要求算法的空间复杂度为O(n)。 
#  你能进一步完善解法吗？要求在C++或任何其他语言中不使用任何内置函数（如 C++ 中的 __builtin_popcount）来执行此操作。 
#  
#  Related Topics 位运算 动态规划 
#  👍 511 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def countBits(self, num: int) -> List[int]:
        # 计数数字的二进制中1的次数
        def bin_count(n):
            count = 0
            while n:
                n = n & (n - 1)
                count += 1
            return count

        res = []
        # 迭代0～n
        for i in range(num + 1):
            res.append(bin_count(i))
        return res
# leetcode submit region end(Prohibit modification and deletion)
