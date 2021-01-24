# 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。 
# 
#  
# 
#  示例 1： 
# 
#  
# 
#  
# 输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
# 输出：6
# 解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
#  
# 
#  示例 2： 
# 
#  
# 输入：height = [4,2,0,3,2,5]
# 输出：9
#  
# 
#  
# 
#  提示： 
# 
#  
#  n == height.length 
#  0 <= n <= 3 * 104 
#  0 <= height[i] <= 105 
#  
#  Related Topics 栈 数组 双指针 动态规划 
#  👍 1959 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def trap(self, height: List[int]) -> int:
        # 没有高度，返回0
        if not height:
            return 0
        n = len(height)
        res = 0
        # 左右边界
        maxleft = [0] * n  # 左边界数组
        maxright = [0] * n  # 右边界数组
        maxleft[0] = height[0]
        maxright[n - 1] = height[n - 1]
        # print(height)
        # tmp = []
        # 左边界数组最大值
        for i in range(1, n):
            maxleft[i] = max(height[i], maxleft[i - 1])
        # print(maxleft)
        # 右边界数组最大值
        for i in range(n - 2, -1, -1):
            maxright[i] = max(height[i], maxright[i + 1])
        # print(maxright)
        # 一次遍历数组，通过高度差得知雨水量
        for i in range(n):
            # tmp.append(min(maxleft[i], maxright[i]))
            if min(maxleft[i], maxright[i]) > height[i]:
                res += min(maxleft[i], maxright[i]) - height[i]
        # print(tmp)
        return res
# leetcode submit region end(Prohibit modification and deletion)
