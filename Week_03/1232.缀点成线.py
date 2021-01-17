#
# @lc app=leetcode.cn id=1232 lang=python3
#
# [1232] 缀点成线
#

# @lc code=start
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        x0 ,y0 = coordinates[0]
        x1 ,y1 = coordinates[1]
        dx , dy = x1-x0 , y1-y0
        for i in range(len(coordinates)):
            temp_dx = coordinates[i][0] - x0
            temp_dy = coordinates[i][1] - y0
            if temp_dx * dy != temp_dy * dx:
                return False
        return True

# @lc code=end

