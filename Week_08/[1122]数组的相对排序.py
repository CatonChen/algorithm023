# 给你两个数组，arr1 和 arr2， 
# 
#  
#  arr2 中的元素各不相同 
#  arr2 中的每个元素都出现在 arr1 中 
#  
# 
#  对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末
# 尾。 
# 
#  
# 
#  示例： 
# 
#  
# 输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
# 输出：[2,2,2,1,4,3,3,9,6,7,19]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= arr1.length, arr2.length <= 1000 
#  0 <= arr1[i], arr2[i] <= 1000 
#  arr2 中的元素 arr2[i] 各不相同 
#  arr2 中的每个元素 arr2[i] 都出现在 arr1 中 
#  
#  Related Topics 排序 数组 
#  👍 159 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        res = []
        # 建立额外数组cnt，记录arri的元素和次数
        cnt = [0] * (max(arr1) + 1)
        for num in arr1:
            cnt[num] += 1
        # 遍历arr2，将arr2中出现的元素加入res
        for num in arr2:
            res.extend([num] * cnt[num])
            cnt[num] = 0  # 使用过的元素次数置为0
        # 遍历不在arr2出现的元素
        for i in range(len(cnt)):
            if cnt[i] > 0:
                res.extend([i] * cnt[i])
        return res
# leetcode submit region end(Prohibit modification and deletion)
