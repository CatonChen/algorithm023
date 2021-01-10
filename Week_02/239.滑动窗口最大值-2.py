#
# @lc app=leetcode.cn id=239 lang=python3
#
# [239] 滑动窗口最大值
#

# @lc code=start
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 使用双端队列，并且存入index
        queue = collections.deque()
        # print(queue)
        res = []
        for i in range(len(nums)):
            # print('for loop i : '+ str(i))
            # 如果当前元素大于单调队列中的尾端元素的话：pop单调队列中的尾端元素
            while queue and nums[queue[-1]] < nums[i]:
                # print('i:'+str(i))
                # print('queue:'+str(queue))
                # print('nums[queue[-1]]:'+str(nums[queue[-1]]))
                # print('nums[i]:'+str(nums[i]))
                queue.pop()
            queue.append(i)
            # print('queue.append(i):'+str(queue))
            # 当单调队列的第一个元素（即最大的元素）不在[i - k + 1, i]，
            # 说明该最大元素在当前的窗口之外，则popleft单调队列中的第一个元素
            if queue[0] <= i - k:
                queue.popleft()
                # print('queue.popleft : '+str(queue))
            # 在当前index >= k - 1的时候（即这时候已经能够构成长度为k的窗口）把单调队列的第一个元素加入到结果中去
            if i >= k - 1:
                res.append(nums[queue[0]])
                # print('res.append(nums[queue[0]]) : ' + str(nums[queue[0]]))
        return res
# @lc code=end

