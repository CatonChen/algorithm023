# 给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。 
# 
#  示例 1: 
# 
#  
# 输入: "aba"
# 输出: True
#  
# 
#  示例 2: 
# 
#  
# 输入: "abca"
# 输出: True
# 解释: 你可以删除c字符。
#  
# 
#  注意: 
# 
#  
#  字符串只包含从 a-z 的小写字母。字符串的最大长度是50000。 
#  
#  Related Topics 字符串 
#  👍 324 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # 定义check判断是否回文字符串
        def check(i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                else:
                    i += 1
                    j -= 1
            return True

        # 判断s是否为回文
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                return check(i + 1, j) or check(i, j - 1)
        # 没有不同的i,j，返回true
        return True
# leetcode submit region end(Prohibit modification and deletion)
