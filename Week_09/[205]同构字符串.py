# 给定两个字符串 s 和 t，判断它们是否是同构的。 
# 
#  如果 s 中的字符可以按某种映射关系替换得到 t ，那么这两个字符串是同构的。 
# 
#  每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。不同字符不能映射到同一个字符上，相同字符只能映射到同一个字符上，字符可以映射到自己本身。 
# 
#  
# 
#  示例 1: 
# 
#  
# 输入：s = "egg", t = "add"
# 输出：true
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "foo", t = "bar"
# 输出：false 
# 
#  示例 3： 
# 
#  
# 输入：s = "paper", t = "title"
# 输出：true 
# 
#  
# 
#  提示： 
# 
#  
#  可以假设 s 和 t 长度相同。 
#  
#  Related Topics 哈希表 
#  👍 334 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        for i in range(len(s)):
            # print(s.index(s[i]))
            # print(t.index(t[i]))
            if s.index(s[i]) != t.index(t[i]):
                return False
        return True

# leetcode submit region end(Prohibit modification and deletion)
