#
# @lc app=leetcode.cn id=21 lang=python3
#
# [21] 合并两个有序链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        prehead=ListNode(0) 
        prev=prehead    #哨兵节点
        while l1 and l2:    #l1和l2都存在时
            if l1.val<l2.val:
                prev.next=l1    #prev指向l1,l1移动一步
                l1=l1.next
            else:
                prev.next=l2
                l2=l2.next
            prev=prev.next
        #当其中一方没有后，链接剩余一方
        prev.next = l1 if l1 is not None else l2

        return prehead.next
# @lc code=end

