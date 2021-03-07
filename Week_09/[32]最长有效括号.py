# 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。 
# 
#  
# 
#  
#  
#  示例 1： 
# 
#  
# 输入：s = "(()"
# 输出：2
# 解释：最长有效括号子串是 "()"
#  
# 
#  示例 2： 
# 
#  
# 输入：s = ")()())"
# 输出：4
# 解释：最长有效括号子串是 "()()"
#  
# 
#  示例 3： 
# 
#  
# 输入：s = ""
# 输出：0
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= s.length <= 3 * 104 
#  s[i] 为 '(' 或 ')' 
#  
#  
#  
#  Related Topics 字符串 动态规划 
#  👍 1197 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        # dp数组
        dp = [0] * n
        for i in range(1, n):  # 从下标1开始，()最短
            if s[i] == ')':
                if s[i - 1] == '(':
                    dp[i] = dp[i - 2] + 2
                # i - dp[i - 1] - 1 >= 0 保证下标有效
                elif s[i - 1] == ')' and i - dp[i - 1] - 1 >= 0 and s[i - dp[i - 1] - 1] == '(':
                    dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2
        # print(dp)
        return max(dp)
# leetcode submit region end(Prohibit modification and deletion)
