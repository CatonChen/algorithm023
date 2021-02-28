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
        if not height:
            return 0
        # 单调栈
        stack = []
        res = 0
        for i in range(len(height)-1):
            # 栈不为空，且height[栈顶]<height[i]
            while len(stack) > 0 and height[stack[-1]] < height[i]:
                top = stack.pop()
                if len(stack) == 0:
                    break
                # 高度
                # print(height[stack[-1]],height[i],height[top])
                h = min(height[stack[-1]], height[i]) - height[top]
                # 宽度
                w = i - stack[-1] - 1
                # 雨水量
                res += (w * h)
            # i入栈
            stack.append(i)
        return res
# leetcode submit region end(Prohibit modification and deletion)
