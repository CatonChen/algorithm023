#
# @lc app=leetcode.cn id=24 lang=python3
#
# [24] 两两交换链表中的节点
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:

        #迭代
        dummyhead=ListNode(0)
        dummyhead.next=head
        temp=dummyhead
        while temp.next and temp.next.next:
            #初始node1 node2
            node1 , node2 = temp.next, temp.next.next
            #两两交换节点
            temp.next = node2   
            node1.next = node2.next
            node2.next = node1
            temp = node1
        return dummyhead.next

# @lc code=end

