# 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。 
# 
#  你可以假设数组是非空的，并且给定的数组总是存在多数元素。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：[3,2,3]
# 输出：3 
# 
#  示例 2： 
# 
#  
# 输入：[2,2,1,1,1,2,2]
# 输出：2
#  
# 
#  
# 
#  进阶： 
# 
#  
#  尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。 
#  
#  Related Topics 位运算 数组 分治算法 
#  👍 843 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # dict = {}
        # #遍历数组，并对元素计数创建对应的字典
        # for num in nums:
        #     if not dict.get(num):
        #         dict[num] = 1
        #     else:
        #         dict[num] += 1
        # # print(dict)
        # tmp = len(nums) // 2
        # # print(tmp)
        # for k, v in dict.items():
        #     # print(k , v )
        #     if v > tmp:
        #         return k

        # # 摩尔投票法
        # major = nums[0]
        # count = 1
        #
        # for i in range(1, len(nums)):
        #     if count == 0:
        #         major = nums[i]
        #         count = 1
        #     else:
        #         if nums[i] == major:
        #             count += 1
        #         else:
        #             count -= 1
        # return major

        # pythonic
        return sorted(nums)[len(nums)//2]
# leetcode submit region end(Prohibit modification and deletion)
