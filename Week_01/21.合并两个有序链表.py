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
        # #递归
        # if l1 is None:      #l1为空返回l2
        #     return l2
        # elif l2 is None:    #l2为空返回l1
        #     return l1     
        # elif l1.val < l2.val:   #判断l1或l2的大小，递归至其中一个完结
        #     l1.next = self.mergeTwoLists(l1.next,l2)
        #     return l1
        # else:
        #     l2.next = self.mergeTwoLists(l1,l2.next)
        #     return l2 

        #迭代
        prehead = ListNode(0)   #空列表
        prev=prehead
        while l1 and l2 :   #l1和l2不为空时
            if l1.val < l2.val: #l1<l2,prev.next指向l1当前值，l1指针后移一位
                prev.next=l1    
                l1=l1.next
            else:               #反之亦然
                prev.next=l2
                l2=l2.next
            prev=prev.next      #prev指针后移一位

        prev.next=l1 if l1 is not None else l2  #prev指针指向剩余的l1或l2
        return prehead.next 
# @lc code=end

