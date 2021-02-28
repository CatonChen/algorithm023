# 给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。 
# 
#  
# 
#  示例： 
# 
#  输入：
# A: [1,2,3,2,1]
# B: [3,2,1,4,7]
# 输出：3
# 解释：
# 长度最长的公共子数组是 [3, 2, 1] 。
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= len(A), len(B) <= 1000 
#  0 <= A[i], B[i] < 100 
#  
#  Related Topics 数组 哈希表 二分查找 动态规划 
#  👍 382 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        # 定义滑动窗口
        def maxLength(addA, addB, length):
            ret = k = 0
            for i in range(length):
                if A[addA + i] == B[addB + i]:
                    # print(i,A[addA+i],B[addB+i])
                    k += 1
                    ret = max(ret, k)
                else:
                    k = 0
            return ret

        m, n = len(A), len(B)
        ret = 0
        for i in range(n):
            #滑动窗口大小
            length = min(m, n - i)
            # print(i,length,maxLength(i,0,length))
            #固定B，滑动A
            ret = max(ret, maxLength(i, 0, length))
        for i in range(m):
            length = min(n, m - i)
            # print(i, length, maxLength(i, 0, length))
            ret = max(ret, maxLength(0, i, length))
        return ret
# leetcode submit region end(Prohibit modification and deletion)
