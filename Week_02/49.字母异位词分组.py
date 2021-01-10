#
# @lc app=leetcode.cn id=49 lang=python3
#
# [49] 字母异位词分组
#

# @lc code=start
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        mp = defaultdict(list)
        for st in strs:
            key = "".join(sorted(st))
            mp[key].append(st) 
        return list(mp.values())
    
# @lc code=end

