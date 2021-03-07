# 给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。 
# 
#  
# 
#  示例： 
# 
#  s = "leetcode"
# 返回 0
# 
# s = "loveleetcode"
# 返回 2
#  
# 
#  
# 
#  提示：你可以假定该字符串只包含小写字母。 
#  Related Topics 哈希表 字符串 
#  👍 354 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def firstUniqChar(self, s: str) -> int:
        # 哈希字典记录字符和次数
        hashmap = {}
        for c in s:
            if hashmap.get(c) is not None:
                hashmap[c] += 1
            else:
                hashmap[c] = 1
        # print(hashmap)
        # 遍历字符串，返回第一个次数为1的元素下标i
        for i in range(len(s)):
            if hashmap[s[i]] == 1:
                return i
        return -1
# leetcode submit region end(Prohibit modification and deletion)
