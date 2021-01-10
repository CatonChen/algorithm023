# WEEK02学习笔记
### 基础知识
#### 哈希表（hash table）
1. 通过关键码值进行查询的数据结构。关键码值（key value）是通过散列函数（hash function）生成的。存放记录的数组叫哈希表，或散列表。
2. 如果一个key value存在多个对应的记录，那么多个记录会通过链表进行存储。
3. 哈希表的应用场景有：
	1. 通讯录，如电话薄，或用户信息（通过uuid保存）
	2. 缓存，如LRU cache
	3. 键值对存储，如redis数据库
4. 哈希表的时间复杂度分析，通常为O(1)；但如果通过链表保存了多个记录，则为O(n)。
5. python常见的数据结构：list、set、dict
#### 树（tree）
1. 树由父节点和子节点构成
2. 链表是特殊化的树。一个父节点只有一个对应的子节点，前后串联起来，就是链表。
3. 树是特殊化的图。因为图有闭环，但树没有闭环。
4. 树的定义语法
```python
#python
class TreeNode:
	def __init__(self, val):
	self.val = val
	self.left, self.right = None, None
```

```java
//Java
public class TreeNode { 
	public int val;
	public TreeNode left, right; 
	public TreeNode(int val) {
		this.val = val; 
		this.left = null; 
		this.right = null;
	} 
}
```

#### 二叉树（binary tree）
1. 二叉树是一种特殊结构的树，每个父节点有且仅有左右两个子节点，如果子节点不为空，可称为完美二叉树。
2. 二叉树的遍历有三种情况：前序，中旬，后序。
	1. 前序（pre-order）：左-\>根-\>右
```python
#python
def preorder(self, root):
	if root: 
		self.traverse_path.append(root.val)
		self.preorder(root.left) 
		self.preorder(root.right)
```
2. 中序（in-order）：根-\>左-\>右
```python
#python
def inorder(self, root):
	if root:
		self.inorder(root.left)
		self.traverse_path.append(root.val)
		self.inorder(root.right)
```
3. 后序（post-order）：左-\>右-\>根
```python
#python
def postorder(self, root):
	if root:
		self.postorder(root.left) 
		self.postorder(root.right)
		self.traverse_path.append(root.val)
```

#### 二叉搜索树（binary search tree）
1. 也称二叉排序树、有序二叉树等，有以下性质特点：
	1. 左子树所有节点的值小于根节点的值
	2. 右子树所有节点的值大于根节点的值
	3. 以此类推，左右子树也分别为二叉搜索树。（重复性！！！）
2. 树的复杂度分析（包含二叉树、二叉搜索树、平衡树、红黑树等）
	1. 时间复杂度：通常O(logn)，最差O(n)
	2. 空间复杂度：一般O(n)
3. 树的排序一般为O(nlogn)，最差可能O(n^2)
4. 树的算法题一般用递归求解

#### HeapSort（堆排序）
堆排序原文ref：[https://www.geeksforgeeks.org/heap-sort/][1]
堆排序是基于二叉堆的数据结构，是一种通过比较大小进行排序的技术。它类似于当做排序查询时，先找出最大的元素，并将其置于尾端。不断地重复这个过程直到所有元素都被排序。
**什么是二叉堆？**
先了解一下完全二叉树。完全二叉树是指除了最底层的节点之外的节点都有两个孩子节点，且最底层的节点尽量靠左的二叉树。
二叉堆是一颗完全二叉树，其所有项按一种特定顺序保存，比如父节点值大于（或小于）其两个子节点值。前者称为最大堆（大顶堆），后者称为最小堆（小顶堆）。堆可以用二叉树或数组表示。
**为什么用数组表示二叉堆？**
因为二叉堆是一个完全二叉树，所以它可以容易的用数组表示，并且数组的空间效率更优。如果存储父节点的下标是I，那左子节点可以通过2\*I+1得到，右子节点可以通过2\*I+2得到（假设数组下标从0开始）。
**升序排序的堆排序算法**[^1]
1. 将输入数据转换成一个大顶堆
2. 此时，最大项保存在堆底部。通过用堆的最后一项替换它，并将堆大小-1。最终，在树的根部建堆。
3. 重复第二步直到堆大小为空。
**如何建立一个堆？**
建堆过程仅能应用于当一个节点及其子节点已经堆化。所以建堆必须是自底向上来进行。
Input data: 4, 10, 3, 5, 1
         4(0)
        /   \
     10(1)   3(2)
    /   \
 5(3)    1(4)

The numbers in bracket represent the indices in the array 
representation of data.
（括号中的数字表示数组中数据的下标）

Applying heapify procedure to index 1:

         4(0)
        /   \
    10(1)    3(2)
    /   \
5(3)    1(4)

Applying heapify procedure to index 0:
        10(0)
        /  \
     5(1)  3(2)
    /   \
 4(3)    1(4)
The heapify procedure calls itself recursively to build heap
 in top down manner.
（建堆过程就是递归地调用自身，自顶向下构建堆）

**python代码实现的堆排序**
```python
# Python program for implementation of heap Sort
 
# To heapify subtree rooted at index i.
# n is size of heap
 
 
def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2
 
    # See if left child of root exists and is
    # greater than root
    if l < n and arr[largest] < arr[l]:
        largest = l
 
    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r
 
    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap
 
        # Heapify the root.
        heapify(arr, n, largest)
 
# The main function to sort an array of given size
 
 
def heapSort(arr):
    n = len(arr)
 
    # Build a maxheap.
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
 
    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)
 
 
# Driver code
arr = [12, 11, 13, 5, 6, 7]
heapSort(arr)
n = len(arr)
print("Sorted array is")
for i in range(n):
    print("%d" % arr[i]),
# This code is contributed by Mohit Kumra
```

### 本周leetcode练习总结
#### 509.斐波那契数列
1. 动态规划
	1. 根据公式f(n)=f(n-1)+f(n-2)的递推关系，利用“滚动数组”的思想解题
	2. 复杂度分析：
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(1)
```python
class Solution:
    def fib(self, n: int) -> int:
        if n<1: return n
        a,b,c=0,0,1
        for i in range(2,n+1):
            a,b=b,c
            c=a+b
        return c
```
3. 更为简洁的写法：
```python
class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return a

作者：ydykid
链接：https://leetcode-cn.com/problems/fibonacci-number/solution/qiu-fei-bo-na-qi-suan-fa-zheng-li-8ge-by-sq8k/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

2. 利用公式，直接递归
```python
class Solution:
    def fib(self, n: int) -> int:
        if n<2:return n 
        return self.fib(n-1)+self.fib(n-2)
```

3. 数学公式计算
```python
# 使用math.pow
class Solution:
    def fib(self, n: int) -> int:
        import math
        return int((math.pow(1 + math.sqrt(5), n) - math.pow(1 - math.sqrt(5), n)) / math.sqrt(5))
# 使用 **替代math.pow
class Solution:
    def fib(self, n: int) -> int:
        return int((((1 + 5**0.5) / 2) ** n - ((1 - 5 ** 0.5) / 2) ** n) / (5**0.5) )

作者：ydykid
链接：https://leetcode-cn.com/problems/fibonacci-number/solution/qiu-fei-bo-na-qi-suan-fa-zheng-li-8ge-by-sq8k/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

#### 350.两个数组的交集
1. 双指针：
	1. 排序两个数组
	2. i,j小于数组长度循环：1）nums1[i]==nums2[j]，把结果放入一个数组，i\++\，j\++；2）nums1[i]\<nums2[j]，i\++；3）反之，j\++
	3. 复杂度分析：
		1. 时间复杂度：O(mlogm) + O(nlogn)；如果m和n已排序，O(min(m,n))
		2. 空间复杂度：O(1)
```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        res=[]
        i=j=0
        while i<len(nums1) and j<len(nums2):
            if nums1[i]==nums2[j]:
                res.append(nums1[i])
                i+=1
                j+=1
            elif nums1[i]<nums2[j]:
                i+=1
            else:
                j+=1
        return res
```
2. 利用collections.counter的特点
	1. 复杂度分析：
		1. 时间复杂度：O(2m+n)
		2. 空间复杂度：O(m+n)
```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        #利用counter
        from collections import Counter
        return [*(Counter(nums1) & Counter(nums2)).elements()]
```

#### 239.滑动窗口最大值（本周复习）
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque	#双端队列
        queue=deque()
        res=[]
        def mover(i):	
            while queue and queue[0]<=i-k :	#当队列存在，当窗口滑动时，将不再窗口中的最大值弹出
                queue.popleft()
            while queue and nums[queue[-1]]<nums[i]:	#当队列存在，且队列尾端元素代表的数组值小于nums[i]值，即该值肯定不是窗口的最大值，弹出队列尾端元素
                queue.pop()
            queue.append(i)

        for i in range(k):	
            mover(i)
        res.append(nums[queue[0]])

        for i in range(k,len(nums)):
            mover(i)
            res.append(nums[queue[0]])
        
        return res     
```

#### 830.较大分组的位置
leetcode国际版，使用python内置库的题解
```python
from itertools import groupby, accumulate

class Solution:
    def largeGroupPositions(self, S: str) -> List[List[int]]:
    	a = list(accumulate([0]+[len(list(i)) for _,i in groupby(S)]))
    	return [[a[i],a[i+1]-1] for i in range(len(a)-1) if a[i+1]-a[i] >= 3]
```

1. 双指针
	1. left是慢指针，index是快指针
	2. 当letter值不等于s[left]的值时，判断index-left\>=3是否成立，并添加结果
	3. 将index的值赋给left
	4. 时间复杂度：O(n)
```python
class Solution:
    def largeGroupPositions(self, S: str) -> List[List[int]]:
        left = 0 
        return_list = []
        S += '1'
        for index, letter in enumerate(S):
            if letter != S[left]:
                if index - left >= 3:
                    return_list.append([left, index - 1])
                left = index
        return return_list
```

#### 1.两数之和（本周复习）
1. 利用字典：从字典中寻找是否已存在（target - val）的值，有则返回该值的下标和val的下标，否则将val的下标添加进字典
2. 时间复杂度：O(n)
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap=dict()
        res=[]
        for i, val in enumerate(nums):
            if hashmap.get(target - val) is not None:
                res.append(hashmap.get(target - val))
                res.append(i)
            hashmap[val]=i
        return res
```

#### 242.有效的字母异位词
1. 利用counter
```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s) == collections.Counter(t) 

作者：JamLeon
链接：https://leetcode-cn.com/problems/valid-anagram/solution/yi-xing-dai-ma-xing-neng-gao-xiao-da-dao-100-by-ja/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

2. 利用字典
	1. 将字母做key，出现的次数做value
	2. 判断两个字典是否有不一致的情况
```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        ds = collections.defaultdict(int)
        dt = collections.defaultdict(int)
        for i in s:
            ds[i]+=1
        for i in t:
            dt[i]+=1
        for i in ds:
            if i not in dt or ds[i]!=dt[i]:
                return False
        return True
```

#### 1021.删除最外层的括号
1. 一次遍历
	1. 类似双指针，但只做一次遍历，遇到最外层括号直接保存，后续返回
	2. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(n)，需要额外空间保存n个元素
```python
class Solution:
    def removeOuterParentheses(self, S: str) -> str:
        counter=0
        res=[]
        for x in S:
            if x=='(' and counter>0 : res.append(x)	#当左括号时，且counter>0，即不是最外层左括号
            if x==')' and counter>1 : res.append(x)	#当右括号时，且counter>1，即不是最外层右括号
            if x=='(' : 
                counter+=1
            else:
                counter-=1
        return "".join(res)
```

2. 双指针
	1. 需要计数器，初始=0，碰到左括号+1，碰到右括号-1，=0时即找到最外层右括号。记录最外层左右括号的下标，之后的左括号又是另一个最外层的左括号。		
	2. 遇到新的左括号，让另一个指针指向新左括号的下标。
	3. 利用数组切片的方法，返回原语。
	4. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(n)，需要额外空间保存
```python
class Solution:
    def removeOuterParentheses(self, S: str) -> str:
        #双指针
        res = []
        left , counter = 0,0
        for i in range(len(S)):
            if S[i]=='(':
                counter+=1
            elif S[i]==')':
                counter-=1
            if counter==0:
                res.append([left,i])
                left=i+1
        # print(res)
        return "".join(S[m+1:n] for m,n in res)     
```

#### 589.N叉树的前序遍历
1. 递归法
	1. 当树为空时返回空
	2. 将树的子节点作为入参传给preorder方法，即通过递归返回所有结果
	3. 复杂度分析
		1. 时间复杂度：O(n)
```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if root is None: return []
        res = [root.val]
        for n in root.children:
            res.extend(self.preorder(n))
        return res
```

2. 递归法（2）
	1. 定义一个函数helper，找出节点值，并继续调用自己找出子节点值
	2. 复杂度分析
		1. 时间复杂度：O(n)
```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res=[]
        def helper(root):
            if not root: return
            res.append(root.val)
            for child in root.children:
                helper(child)
        helper(root)
        return res
```

3. 迭代法
	1. 定义一个栈，初始保存根节点值
	2. 从栈顶取出节点，保存该节点值，并把该节点的子节点逆序压入栈中，循环至栈为空为止。
	3. 复杂度分析
		1. 时间复杂度：O(n)
```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root: return []	#树为空，返回空
        res=[]	
        st = [root]	#把root压入栈，作为第一个
        while st:	#当栈不为空循环
            node = st.pop()	#弹出栈顶元素
            res.append(node.val)			
            st.extend(node.children[::-1])	#将该节点的子节点逆序压入栈中
        return res
```

#### 412.Fizz Buzz
1. 迭代解法
	1. n%3==0，给Fizz; n%5==0，给Buzz; 前两者都成立给Fizz+Buzz
	2. 一次遍历，1～n+1后，返回结果
	3. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(n)，利用额外空间保存结果，不考虑额外空间即O(1)
```python
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        res=[]
        for i in range(1,n+1):
            num_str=""
            divi_3=(i%3==0) #i%3==0,即整除3
            divi_5=(i%5==0) #i%5==0,即整除5
            if divi_3:  num_str+="Fizz"
            if divi_5:  num_str+="Buzz"
            if not num_str: num_str=str(i)
            res.append(num_str)
        return res
```

2. 哈希表(hash)
	1. 定义字典，dict=\{3:’Fizz’, 5:’Buzz’}，优势是方便后续增加新的字符串
	2. 遍历1～n+1，i%dict.key==0，即i可以被3或5整除，拼接字符串，最后返回结果
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历1～n
		2. 空间复杂度：O(n)，利用额外空间保存结果，不考虑额外空间即O(1)
```python
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        res=[]
        ditc={3:"Fizz", 5:"Buzz"}
        for i in range(1,n+1):
            num_str=""
            for key in ditc.keys(): #遍历字典中key值
                if i%key==0: num_str+=ditc[key] #i%key==0，即符合条件
            if not num_str: num_str=str(i)
            res.append(num_str)
        return res
```

3. 一行代码（国际版解法）
	1. 将多个条件判断放在一行中，边判断边输出返回值
	2. 复杂度分析：
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(1)
```python
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
    	return ['FizzBuzz' if i%15 == 0 else 'Buzz' if i%5 == 0 else 'Fizz' if i%3 == 0 else str(i) for i in range(1,n+1)]
```

#### 258.各位相加
1. 数学推导
	1. 寻找数学规律，“任意数为9的倍数时，其位数最终和必为0”的规律。
	2. 时间复杂度：O(1)
```python
class Solution:
    def addDigits(self, num: int) -> int:
        if num==0: return 0
        if num%9==0: return 9
        return num%9
```

2.  数学推导代码优化
```python
class Solution:
    def addDigits(self, num: int) -> int:
        if (num%9==0) and (num!=0):
            return 9
        else:
            return num%9
```

3. 循环迭代
	1. 将num变为数组，然后循环相加
	2. 当循环相加之和\>=10，就反复进行第一步的做法，直到相加之和为个位数返回
	3. 复杂度分析：
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(1)
```python
class Solution:
    def addDigits(self, num: int) -> int:
        while num >=10:
            digits = list(map(int,str(num)))
            num=0
            for i in range(len(digits)):
                num+=digits[i]
        return num
```

#### 189.旋转数组（本周复习）
1. 双指针迭代，左右夹逼，交换数组位置
2. 根据k%len(nums)的余数，切分数组，总共迭代三次，分别为（0,n-k-1）；（n-k,n-1）;（0,n-1）。
3. 复杂度分析
	1. 时间复杂度：O(n)
	2. 空间复杂度：O(1)
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        #双指针，两边夹逼，交换数组位置
        #左右指针位置，根据k%len(nums)来定
        n=len(nums)
        k=k%n
        def swap(i,j):
            while (i<j):
                nums[i],nums[j]=nums[j],nums[i]
                i+=1
                j-=1
        swap(0,n-k-1)
        swap(n-k,n-1)
        swap(0,n-1)
```

2. 数组切片
	1. 根据k%n的余数，拼接数组nums[:n-k]+nums[n-k:]
	2. 复杂度分析
		1. 时间复杂度：O(1)
		2. 空间复杂度：O(1)
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)
        k=k%n
        nums[:] = nums[n-k:]+nums[:n-k]
```

#### 104.二叉树的最大深度
1. 递归法
	1. 规则：找出左子树l和右子树r，其中最大深度max(l,r)+1，即二叉树的最大深度。而左子树或右子树的最大深度，可以复用这个规则找出，一直到没有子树的子节点为止。
	2. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(n)，n为树高
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None: 
            return 0    #根节点为空，即0
        else:
            l_h = self.maxDepth(root.left)  #递归根节点的左子树
            l_r = self.maxDepth(root.right) #递归根节点的右子树
            return max(l_h,l_r)+1   
```

2. 迭代解法bfs（广度优先搜索）
	1. 遍历二叉树的每一层，最后返回层数
	2. 利用双端队列deque()保存每一层的节点，直到没有下一层节点为止，每循环一层，计数器+1
	3. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(n)，利用了额外空间
```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        from collections import deque
        if root is None:
            return 0
        queue = deque([root])	#把根节点放入队列
        level = 0 
        while queue:	#当queue不为空
            n = len(queue)	
            for i in range(n):	
                node = queue.popleft()	#循环弹出每一层已保存的节点，如果弹出节点仍有左右子节点，则放入队列下次循环
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            level+=1	#当本层所有节点弹出后，层数+1
        return level
```
4. 另一种写法，使用extend方法，缩减了对node子节点的判断逻辑
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        from collections import deque
        if root is None:
            return 0
        queue = deque([root])
        level = 0 
        while queue:
            n = len(queue)
            for i in range(n):
                node = queue.popleft()
                if node:	
                    queue.extend([node.left,node.right])
            level+=1
        return level -1
```
5. 上面两个题解在返回上的差异分析：
    3
   / \
  9  20
    /  \
   15   7

题解一，deque打印如下：
第一次pop队列的元素：
队列长度n=1
' +
  'TreeNode{val: 3, left: TreeNode{val: 9, left: None, right: None}, right: TreeNode{val: 20, left: TreeNode{val: 15, left: None, right: None}, right: TreeNode{val: 7, left: None, right: None}}}
' +
第二次pop队列的元素
队列长度n=2
' +
  'TreeNode{val: 9, left: None, right: None}
' +
  'TreeNode{val: 20, left: TreeNode{val: 15, left: None, right: None}, right: TreeNode{val: 7, left: None, right: None}}
' +
第三次pop队列的元素
队列长度n=2
' +
  'TreeNode{val: 15, left: None, right: None}
' +
  'TreeNode{val: 7, left: None, right: None}
一共循环了三次，层次level=3，返回不需要-1

题解二，deque打印如下：
第一次pop队列元素：
队列长度n=1
' +
  'TreeNode{val: 3, left: TreeNode{val: 9, left: None, right: None}, right: TreeNode{val: 20, left: TreeNode{val: 15, left: None, right: None}, right: TreeNode{val: 7, left: None, right: None}}}
' +
第二次pop队列元素：
队列长度n=2
' +
  'TreeNode{val: 9, left: None, right: None}
' +
  'TreeNode{val: 20, left: TreeNode{val: 15, left: None, right: None}, right: TreeNode{val: 7, left: None, right: None}}
' +
第三次pop队列元素
队列长度n=4
' +
  'None
' +
  'None
' +
  'TreeNode{val: 15, left: None, right: None}
' +
  'TreeNode{val: 7, left: None, right: None}
' +
第四次pop队列元素
队列长度n=4
' +
  'None
' +
  'None
' +
  'None
' +
  'None
一共循环了四次，层次level=4，所以返回时-1
造成这个差异点分析：
1.主要是因为判断条件不同，与append和extend方法实现差异的无关
2.题解一，是对当前节点是否继续存在有效左右子节点判断后再添加进队列；
3.题解二，是对当前节点是否有效，而不是左右子节点是否有效做判断；后续直接将子节点加入队列，造成队列循环多了一次。

3. 使用stack（栈）迭代
	1. 与使用deque队列类似，当存在子节点时，将子节点压入栈
	2. 不同点：a.将节点压入栈时，同时记录高度；b.取出栈顶节点时，要比较当前高度与栈顶节点的高度，返回较大者
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        stack,depth = [(root,1)],0
        while stack:
            cur,cur_depth = stack.pop()
            if cur:
                depth = max(depth,cur_depth)
                if cur.left:
                    stack.append((cur.left,cur_depth+1))
                if cur.right:
                    stack.append((cur.right,cur_depth+1))
        return depth
```

#### 49.字母异位词分组
1. 排序
	1. 字母异位词，指其字母相同，字母顺序不同。利用这个特点，可以将其排序后的相同值作为key保存在哈希表中，哈希表的值则为一组字母一位词。
	2. 复杂度分析
		1. 时间复杂度：O(blogs)，考虑两点：a.给定字符串中的异位词个数；b.每个异位词的长度
		2. 空间复杂度：O(n)，需要额外空间保存，异位词个数\*每个异位词长度
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        mp = defaultdict(list)
        for st in strs:
            key = "".join(sorted(st))
            mp[key].append(st) 
        return list(mp.values())
```

2. 质数运算
	1. 思路：利用质数找出每个异位词的key，然后将相同key的异位词放在一组里，最后返回字典的value。
	2. 如何理解质数计算？
```python
class Solution:
    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
    def ks(s):
        c = 1
        for cc in s: c *= Solution.prime[ord(cc)-97]
        return c
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = {}
        for s in strs:
            k = Solution.ks(s)
            if k not in res: res[k] = [s]
            else: res[k].append(s)
        return list(res.values())

作者：kknike
链接：https://leetcode-cn.com/problems/group-anagrams/solution/jiu-shi-bi-pin-wei-yi-keyde-kuai-man-by-8l2go/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

3. 素数计算
	1. 利用字典保存每个字母的素数（与2.质数运算思路一样，但数字选择不一样，这里不太理解…）
	2. 遍历每个异位词的字母，所有字母相乘的值是唯一的，将该值作为字典的key，将同样key的异位词作为字典的值
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        charl = {'a':2,'b':3,'c':5,'d':7,'e':11,'f':13,'g':17,'h':19,'i':23,
             'j':29,'k':31,'l':37,'m':41,'n':43,'o':47,'p':53,'q':59,'r':61,
             's':67,'t':71,'u':73,'v':79,'w':83,'x':89,'y':97,'z':101}
        hashAllChars = {}
        for i in strs:
            r = 1
            for j in i:
                r *= charl[j]
            if r in hashAllChars:
                hashAllChars[r].append(i)
            else:
                hashAllChars[r] = [i]
        return hashAllChars.values()
```

#### 102.二叉树的层序遍历
1. 广度优先搜索（bfs）
	1. bfs使用队列，把没有搜索到的点依次放入队列，然后再弹出队列头部元素做当前遍历点。bfs有两个模板：
		1. 不需要记录当前遍历到哪一层
```python
while queue 不空：
    cur = queue.pop()
    for 节点 in cur的所有相邻节点：
        if 该节点有效且未访问过：
            queue.push(该节点)

作者：fuxuemingzhu
链接：https://leetcode-cn.com/problems/binary-tree-level-order-traversal/solution/tao-mo-ban-bfs-he-dfs-du-ke-yi-jie-jue-by-fuxuemin/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
2. 需要记录当前遍历到哪一层。增加level表示当前遍历到二叉树哪一层了；size表示当前遍历层有多少元素，当size为空，即遍历完当前层的所有元素。
```python
level = 0
while queue 不空：
    size = queue.size()
    while (size --) {
        cur = queue.pop()
        for 节点 in cur的所有相邻节点：
            if 该节点有效且未被访问过：
                queue.push(该节点)
    }
    level ++;

作者：fuxuemingzhu
链接：https://leetcode-cn.com/problems/binary-tree-level-order-traversal/solution/tao-mo-ban-bfs-he-dfs-du-ke-yi-jie-jue-by-fuxuemin/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
2. 以上两个模板是通用模板，本题采用模板二
	3. 使用队列保留每层所有节点，每次把原先节点进行出队列操作，再把每个节点非空左右子节点添加进队列。最终得到每层的遍历。
```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        from collections import deque
        queue = deque()
        queue.append(root)
        res=[]
        while queue:
            n=len(queue)
            level = []
            for i in range(n):
                cur=queue.popleft()
                if cur is None: 
                    continue
                level.append(cur.val)
                queue.append(cur.left)
                queue.append(cur.right)
            if level:
                res.append(level)
        return res
```

2. 深度优先搜索（dfs）
	1. 先递归左子树，再递归右子树
	2. 为保证递归过程中同一层的节点放入同一个列表中，需要记录节点的深度level，遇到新节点将其放入对应level的列表末端。
	3. 遍历到新level时，如果最终结果res中没有对应的level，需要创建。
```python
class Solution:
    def helper(self,root,level,res):
        if root is None: return
        if len(res)==level:
            res.append([])
        res[level].append(root.val)
        if root.left:
            self.helper(root.left,level+1,res)
        if root.right:
            self.helper(root.right,level+1,res)
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        self.helper(root,0,res)
        return res
```

#### 94.二叉树的中序遍历
1. 颜色标记法（leetcode中国版精选）：兼具栈迭代的高效，又像递归一样简洁易懂，更重要的是，这种方法对于前序、中序和后序可以写出完全一致的代码（调整左右子节点入栈顺序）。
	1. 使用颜色标记节点状态，新节点标记白色，已访问的标记灰色
	2. 如遇到的节点为白色，将其标记为灰色，之后将右子节点、自身和左子节点按顺序入栈（中序：左-\>根-\>右；因为栈先入后出，所以进栈顺序右-\>根-\>左。）
	3. 如遇到的节点为灰色，则返回其值
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        white,grey=0,1
        stack=[(white,root)]
        res=[]
        while stack:
            color,node=stack.pop()
            if node is None: continue
            if color==white:
                stack.append((white,node.right))
                stack.append((grey,node))
                stack.append((white,node.left))
            else:
                res.append(node.val)
        return res
```

2. 递归法：按树的中序遍历调用自身即可
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None: return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

#### 283.移动零（本周复习）
1. 迭代
	1. 初始num\_zero=0，用于标识第一个0的下标。
	2. 遍历数组nums，当下标i的元素不等于0时，交换i和num\_zero的元素位置，交换后num\_zero向右移动一步，直至数组遍历完毕。
	3. 复杂度分析
		1. 时间复杂度：O(n)，n指数组nums的长度
		2. 空间复杂度：O(1)
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        num_zero=0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[num_zero],nums[i]=nums[i],nums[num_zero]
                num_zero+=1
```

#### 429.N叉树的层序遍历
1. 队列bfs
	1. 使用deque保存每一层的节点数。
	2. 使用while循环来处理当前层的节点，使用for循环处理每个节点。
	3. for循环中，逐步弹出当前层的节点，将其节点值存储在临时数组中，并将其子节点放入队列，用于下次循环
	4. 当一次for循环结束后，将临时数组中的值添加进结果数组，然后进行下一次while循序，直至queue为空。最后返回结果数组
```python
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if root is None:
            return []
        res = []
        queue = collections.deque()
        queue.append(root)	#将根节点加入队列
        while queue:	#当队列不为空循环
            level=[]
            for i in range(len(queue)):
                node = queue.popleft()	#弹出头部元素
                level.append(node.val)	#保存节点值
                queue.extend(node.children)	#将子节点加入队列用于下次循环
            res.append(level)
        return res
```

[^1]:	此处翻译可能存在理解不准确

[1]:	https://www.geeksforgeeks.org/heap-sort/