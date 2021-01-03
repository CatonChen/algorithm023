## WEEK01学习笔记

#### 基础知识
##### 数组 array
_时间复杂度_
- 查询：O(1)
- 插入、删除：O(n)
- 头节点、尾节点添加：O(1)头节点添加O(1)，需在数组头部提前预留空间，将下标前移一个位置插入元素。
##### 链表 linked\_list
_时间复杂度_
- 头节点、尾节点添加：O(1)
- 插入、删除：O(1)
- 查询（非头、尾节点）：O(n)
- 查询（头、尾节点）：O(1)
##### 跳表 skip\_list
> 1. 必须是有序的链表
> 2. 对有序链表进行升维，增加一级或多级索引，降低时间复杂度成本
> 3. 增加了索引维护成本
_时间复杂度_
- 插入、删除、查询：O(logn)
##### 栈 stack
1. 后进先出（先入后出）
 _时间复杂度_
- 插入、删除：O(1)
- 查询：O(n)
##### 队列 queue
1. 先入先出
_时间复杂度_
- 插入、删除：O(1)
- 查询：O(n)
##### 双端队列 deque
1. 栈和队列的结合体，前后都可以添加、删除元素
_时间复杂度_
- 插入、删除：O(1)
- 查询：O(n)
2. [python双端队列参考][1]
	1. add first 可以用appendleft()方法
	2. add last 仍然用append()方法
##### 优先队列 priority queue
1. 是一种抽象的数据结构，底层实现方式很多样复杂
2. 常用heap（堆）来实现优先队列
_时间复杂度_
- 插入：O(1)
- 删除：O(logn)  #按元素优先级取出
3. [python 优先队列参考][2]
#### LeetCode本周练习总结
##### 283. 移动零
1. 双指针题解：
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
		#定义i,j两个指针
        i = 0
        j = 1
        
        while j < len(nums):	#当j小于数组长度时循环
            if nums[i] == nums[j] and nums[i] == 0:
			#当i,j的元素为0时，移动j
                j += 1
                continue
            
            if nums[i] == 0 and nums[j] != 0:
			#当j的元素不为0时，交换元素位置
                # swap elements
                temp = nums[j]
                nums[j] = nums[i]
                nums[i] = temp
            #i,j两个指针各移动一步
            i += 1
            j += 1
```
2. 数组位置交换题解：
```python
def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)	#数组长度
        non_zero = 0	
        
        for i in range(n):	#遍历数组
            if nums[i] != 0:	#当i的元素不为0时
				#交换i和non_zero的位置
                nums[non_zero],nums[i] = nums[i],nums[non_zero]
                non_zero +=1	#交换后non_zero移动一步
```
##### 66.加一
1. 将数组转为string，再转为int然后+1，将+1结果转为string，再转为int，再用map方法变为数组
2. 返回结果时，在前面补0
3. 时间复杂度O(n)
```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        r = list(map(int,str(int(''.join(map(str,digits)))+1)))
        return [0]*(len(digits) - len(r)) + r
```
##### 70.爬楼梯
1. 动态规划
2. f(n)=f(n-1)+f(n-2)   斐波那契数列思想
3. 迭代法题解
	1. 空间复杂度O(1)，时间复杂度O(n)
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n==1 or n==2 : return n	#当n为1或2时，返回n
        a,b,c=1,2,0
        for i in range(3,n+1):	#i从3开始循环至n+1
            c=a+b	#c为a,b之和
            a=b		
            b=c		
        return c	#返回c最后的结果
```
4. 斐波那契数列题解
	1. 如何理解黄金分割比？
	2. 时间、空间复杂度O(1)  --最效率的解法
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        import math
        sqrt5=5**0.5	#黄金分割比
        fibin=math.pow((1+sqrt5)/2,n+1)-math.pow((1-sqrt5)/2,n+1)
        return int(fibin/sqrt5)
```
##### 26.删除排序数组中的重复项
1. 快慢指针
2. 当快指针元素不等于慢指针元素时，慢指针移动1步后，将快指针元素赋值给慢指针元素。
3. 直到快指针遍历完整个数组，返回慢指针+1
```python
input = [0,1,1,1,2,2,3,3,4]
tail                 fast            nums
1                     1      [0, 1, 1, 1, 2, 2, 3, 3, 4]
1                     2      [0, 1, 1, 1, 2, 2, 3, 3, 4]
1                     3      [0, 1, 1, 1, 2, 2, 3, 3, 4]
2                     4      [0, 1, 2, 1, 2, 2, 3, 3, 4]
2                     5      [0, 1, 2, 1, 2, 2, 3, 3, 4]
3                     6      [0, 1, 2, 3, 2, 2, 3, 3, 4]
3                     7      [0, 1, 2, 3, 2, 2, 3, 3, 4]
4                     8      [0, 1, 2, 3, 4, 2, 3, 3, 4]

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 0 
        if len(nums)==0:
            return 0
        for i in range(1,len(nums)): #指针位移
            if nums[l]!=nums[i]:    #两者不等
                l+=1                #移动第一个指针
                nums[l]=nums[i]     #赋值
        return l+1                  #返回长度
```
##### 1.两数之和
1. 暴力解法
	1. 双指针，内外循环
	2. 将符合条件的下标存储到数组中
	3. 时间复杂度O(i\*j)，有大量重复计算容易超时
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        res=[]	#创建一个空数组
        for i in range(n):
            for j in range(i+1,n):
                if nums[i]+nums[j]==target:	#符合条件的下标添加到数组
                    res.append(i)
                    res.append(j)
        return res
```
2. 字典解法
	1. 判断字典中是否存在符合条件的元素：
		1. 若有，返回已保存的下标，和当前i代表的下标
		2. 若无，将当前i的元素和下标i保存进字典
	2. 时间复杂度O(n)，只需遍历一次数组
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap=dict()
        for i , v in enumerate(nums):
            if hashmap.get(target - v) is not None:
                return [hashmap.get(target - v),i]
            hashmap[v]=i
```
21.合并两个有序链表
1. 递归解法
	1. 如果l1或l2其中一个为空，返回另一个
	2. 如果l1.val小于l2.val，则l1.next指向l2；反之亦然
	3. 时间、空间复杂度均为O(m+n)
```python
class Solution:
    def mergeTwoLists(self, l1, l2):
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```
2. 迭代解法
	1. 增加一个哨兵节点，通过哨兵节点重新合并l1和l2
	2. 当l1.val\<=l2.val时，prev.next指向l1，l1=l1.next；反之亦然；最后prev指向prev.next
	3. l1或l2循环结束后，prev.next指向剩下的l1或l2
	4. 最终返回新链表prehead.next的部分
```python
class Solution:
    def mergeTwoLists(self, l1, l2):
        prehead = ListNode(-1)	#新链表
        prev = prehead		#哨兵节点
        while l1 and l2:	#l1,l2都不为空时
            if l1.val <= l2.val:	
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next            
            prev = prev.next

        # 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev.next = l1 if l1 is not None else l2
        return prehead.next
```
##### 1046.最后一块石头的重量
使用堆heapq 解法
1. heapq默认最小堆，通过负数变为最大堆
	2. 当堆\>1时遍历，连续弹出两个最大值赋给x,y
	3. 当x,y不等时，往堆中插入x-y的值
	4. 遍历后，当堆存在时，返回堆的第一个元素值的负数，否则返回0
```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        # 初始化
        heap = [-stone for stone in stones]
        heapq.heapify(heap)

        # 模拟
        while len(heap) > 1:
            x,y = heapq.heappop(heap),heapq.heappop(heap)
            if x != y:
                heapq.heappush(heap,x-y)

        if heap: return -heap[0]
        return 0
```
##### 189.旋转数组
1. 迭代解法
	1. k=k%n  —取余数，这里k=3，根据余数切分数组
	2. 0-k ，[0,3] == [1,2,3,4] , k+1 - n [4,6] == [5,6,7]
	3. 当 i=0 ，j =3 时，交换 nums[i] ,nums[j]的位置， i+=1 ,j-=1
	4. [1,2,3,4] -\> [4,3,2,1] , [5,6,7] -\> [7,6,5]
	5. 整体交换 
```python
输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k%n	#取余数，后续对整体数组进行分割使用
        def swap(i,j):	#定义swap函数
            while (i<j):	#i<j时，交换i,j位置的元素，i++,j--
                nums[i],nums[j] = nums[j],nums[i]
                i+=1
                j-=1
		#调用三次swap函数
        swap(0,n-k-1)
        swap(n-k,n-1)
        swap(0,n-1)
```
2. 切片
	1. 充分利用了python数组特性，简单易懂
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n=len(nums)
        k%=n
        nums[:]=nums[n-k:]+nums[:n-k]
```
3. 利用（双端队列）collections.deque.rotate方法实现？
##### 239.滑动窗口最大值
_本题还需多理解题解_
1. 利用单调队列的有序性，判断队列尾端元素和i当前元素的大小，来决定是否弹出尾端元素
2. 如何理解单调队列的第一个元素不在[i-k+1,i]中？
```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        # 使用双端队列，并且存入index
        queue = collections.deque()
        res = []
        for i in range(len(nums)):
            # 如果当前元素大于单调队列中的尾端元素的话：pop单调队列中的尾端元素
            while queue and nums[queue[-1]] < nums[i]:
				#print('i:'+str(i))
                #print('queue:'+str(queue))
                #print('nums[queue[-1]]:'+str(nums[queue[-1]]))
                #print('nums[i]:'+str(nums[i]))
				#打印结果看下面
                queue.pop()
            queue.append(i)
            # 当单调队列的第一个元素（即最大的元素）不在[i - k + 1, i]，
            # 说明该最大元素在当前的窗口之外，则popleft单调队列中的第一个元素
            if queue[0] <= i - k:
                queue.popleft()
            # 在当前index >= k - 1的时候（即这时候已经能够构成长度为k的窗口）把单调队列的第一个元素加入到结果中去
            if i >= k - 1:
                res.append(nums[queue[0]])
        return res


#例如：nums = [1,3,-1,-3,5,3,6,7], k = 3
#关键步骤打印结果
```
' 'for loop i : 0
' +
  'queue.append(i):deque([0])
' +
  'for loop i : 1
' +
  'i:1
' +
  'queue:deque([0])
' +
  'nums[queue[-1]]:1
' +
  'nums[i]:3
' +
  'queue.append(i):deque([1])
' +
  'for loop i : 2
' +
  'queue.append(i):deque([1, 2])
' +
  'res.append(nums[queue[0]]) : 3
' +
  'for loop i : 3
' +
  'queue.append(i):deque([1, 2, 3])
' +
  'res.append(nums[queue[0]]) : 3
' +
  'for loop i : 4
' +
  'i:4
' +
  'queue:deque([1, 2, 3])
' +
  'nums[queue[-1]]:-3
' +
  'nums[i]:5
' +
  'i:4
' +
  'queue:deque([1, 2])
' +
  'nums[queue[-1]]:-1
' +
  'nums[i]:5
' +
  'i:4
' +
  'queue:deque([1])
' +
  'nums[queue[-1]]:3
' +
  'nums[i]:5
' +
  'queue.append(i):deque([4])
' +
  'res.append(nums[queue[0]]) : 5
' +
  'for loop i : 5
' +
  'queue.append(i):deque([4, 5])
' +
  'res.append(nums[queue[0]]) : 5
' +
  'for loop i : 6
' +
  'i:6
' +
  'queue:deque([4, 5])
' +
  'nums[queue[-1]]:3
' +
  'nums[i]:6
' +
  'i:6
' +
  'queue:deque([4])
' +
  'nums[queue[-1]]:5
' +
  'nums[i]:6
' +
  'queue.append(i):deque([6])
' +
  'res.append(nums[queue[0]]) : 6
' +
  'for loop i : 7
' +
  'i:7
' +
  'queue:deque([6])
' +
  'nums[queue[-1]]:6
' +
  'nums[i]:7
' +
  'queue.append(i):deque([7])
' +
  'res.append(nums[queue[0]]) : 7
3. 老师提供的解法
```python
    def maxSlidingWindow(self, nums, k):
        win = [] # 在window里的元素的index
        ret = []
        for i, v in enumerate(nums):
            if i >= k and win[0] <= i - k: win.pop(0)	#这个条件的含义？
            while win and nums[win[-1]] <= v: win.pop()
            win.append(i)
            if i >= k - 1: ret.append(nums[win[0]])
        return ret
```
##### 88.合并两个有序数组
1. 利用数组本身的特点
	1. 合并数组，重新排序
	2. 时间、空间复杂度最高
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums1[:]=nums1[:m]+nums2
        # print(nums1)
        nums1.sort()
        # print(nums1)
```
2. 迭代法
	1. 从后往前倒推；k为nums1的整个长度（因nums1有足够空间保存nums2）；i是nums1的下标，j是nums2的下标
	2. 当i的元素\>j的元素时，k元素=i元素值，反之亦然；k--
	3. 如果j\>i，当循环结束后，将nums2剩余的元素补到nums1里
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i, j , k = m-1 ,n-1 ,m+n-1
        while i>=0 and j>=0 :
            if nums1[i]>nums2[j]:
                nums1[k]=nums1[i]
                i-=1
            else:
                nums1[k]=nums2[j]
                j-=1
            k-=1

        nums1[:j+1]=nums2[:j+1]
```
##### 641.设计循环双端队列
1. 数组解法，利用数组的特点，易于理解
```python
class MyCircularDeque:

	def __init__(self, k: int):
	    """
	    Initialize your data structure here. Set the size of the deque to be k.
	    """
	    self.cq = []*k        
	    self.size = 0		#数组计数器
	    self.max_len = k	#数组最大长度
	    
	def insertFront(self, value: int) -> bool:
	    """
	    Adds an item at the front of Deque. Return true if the operation is successful.
	    """
	    if(self.size == self.max_len):
	        return False
	    self.cq.insert(0, value)	#在数组头部添加，计数器+1
	    self.size+=1
	    return True
	
	def insertLast(self, value: int) -> bool:
	    """
	    Adds an item at the rear of Deque. Return true if the operation is successful.
	    """
	    if(self.size == self.max_len):
	        return False
	    self.cq.append(value)	#添加进数组尾端，计数器+1
	    self.size+=1
	    return True
	
	def deleteFront(self) -> bool:
	    """
	    Deletes an item from the front of Deque. Return true if the operation is successful.
	    """
	    if(not self.size):
	        return False
	    self.cq.pop(0)	#弹出数组第一个元素，计数器-1
	    self.size-=1
	    return True
	
	def deleteLast(self) -> bool:
	    """
	    Deletes an item from the rear of Deque. Return true if the operation is successful.
	    """
	    if(not self.size):
	        return False
	    self.cq.pop()	#弹出数组尾端元素，计数器-1
	    self.size-=1
	    return True
	
	def getFront(self) -> int:
	    """
	    Get the front item from the deque.
	    """
	    return -1 if not self.size else self.cq[0]	#取数组第一个元素
	
	def getRear(self) -> int:
	    """
	    Get the last item from the deque.
	    """
	    return -1 if not self.size else self.cq[-1]		#取数组尾端元素
	
	def isEmpty(self) -> bool:
	    """
	    Checks whether the circular deque is empty or not.
	    """
	    return not self.size	#计数器无值
	
	def isFull(self) -> bool:
	    """
	    Checks whether the circular deque is full or not.
	    """
	    return self.size == self.max_len	#计数器等于数组最大长度
```
##### 42.接雨水
1. 双指针：
	1. 利用双指针计算**每一最低高度的面积**，累加可得接满雨水后的总面积数
	2. 寻找每个高度范围：
		1. temp\_h：移动双指针前，双指针处的最小高度
		2. min\_l\_f：移动双指针后，双指针处的最小高度
	3. 若移动双指针后，双指针处的最小高度有变化min\_l\_f\>temp\_h，则将增加的高度（min\_l\_f - temp\_h）乘以对应高度的范围（right - left +1）的面积添加到总面积中【result += (min\_l\_f-temp\_h) \* (right - left + 1)】
	4. 双指针不断向中间移动，将改变的面积添加到总面积中
	5. 最后减去全部柱子的面积数，即可得到雨水面积
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        result = 0
        left, right = 0, len(height) - 1	#左右边界的柱子
        temp_h = 0	#用于保存上次左右边界柱子中高度较小的柱子
        while left <= right:
            min_l_f = min(height[left], height[right])	#找出左右边界柱子中高度较小的柱子
            if min_l_f > temp_h:
                result += (min_l_f-temp_h) * (right - left + 1)	#高度差*对应范围
                temp_h = min_l_f
            while left<=right and height[left] <= temp_h:	#当左边边界柱子高度小于temp_h，移动左边柱子
                left += 1
            while left<=right and height[right] <= temp_h:	
                right -= 1
        result = result-sum(height)	#总面积-柱子面积，得到雨水面积
        return result
```

[1]:	https://docs.python.org/zh-cn/3/library/collections.html?highlight=deque#deque-objects
[2]:	https://docs.python.org/zh-cn/3/library/heapq.html#priority-queue-implementation-notes