# WEEK08学习笔记
### 基础知识
#### 平衡二叉树（AVL树）
- 平衡二叉树具有普通二叉树的所有特性。相对于普通二叉树，多了平衡因子。
- 平衡因子是它的左子树高度减去右子树高度（有时相反）。
- 平衡因子balancefactor=\{-1,0,1}。
- 节点需要需要额外空间存储平衡因子，且调整频繁。
平衡二叉树通过旋转操作来保存平衡性。四种旋转操作分别为左旋、右旋、左右旋和右左旋。
- 左旋：通常应用于右右子树；具体为子树的第二个节点提上去做根节点，第一个节点拉下来做左子节点。子树变为高度一致的平衡树。
- 右旋：通常应用于左左子树；具体为子树的第二个节点提上去做根节点，第一个节点拉下来做右子节点。子树变为高度一致的平衡树。
- 左右旋：通常应用于左右子树；具体操作为，首先通过左旋将左右子树转换为左左子树，再通过右旋将左左子树变为平衡树。
- 右左旋：通常应用于右左子树；具体操作为，首先通过右旋将右左子树转换为右右子树，再通过左旋将右右子树变为平衡树。
#### 红黑树
红黑树是一种近似平衡的二叉搜索树(Binary Search Tree)，它能够确保任何一 个结点的左右子树的高度差小于两倍。
具体来说，红黑树是满足如下条件的二叉搜索树:
- 每个结点要么是红色，要么是黑色
- 根结点是黑色
- 每个叶结点(NIL结点，空结点)是黑色的。
- 不能有相邻接的两个红色结点
- 从任一结点到其每个叶子的所有路径都包含相同数目的黑色结点。
_**从根到叶子的最长的可能路径不多于最短的可能路径的两倍长。**_
#### 平衡二叉树与红黑树的对比
- AVL trees provide faster lookups than Red Black Trees because they are more strictly balanced.（AVL树比红黑树提供更快的查询，但有更严格的平衡性要求。）
- Red Black Trees provide faster insertion and removal operations than AVL trees as fewer rotations are done due to relatively relaxed balancing.（红黑树相对于AVL树具有更灵活的平衡性，因此较少的选择操作提供了更快的插入与删除操作。）
- AVL trees store balance factors or heights with each node, thus requires storage for an integer per node where as Red Black Tree requires only 1 bit of information per node.（AVL树的每个节点都需要一个整型的存储空间保存平衡因子或高度，但红黑树的每个节点只需要1字节保存额外信息。）
- Red Black Trees are used in most of the language libraries like map, multimap, multiset in C\+\+ where as AVL trees are used in databases where faster retrievals are required.（红黑树多见于语言库，比如C\+\+的map、multimap和multiset函数等；同理AVL树多用于数据库，因为需要更快的检索功能。）

#### 位运算
为什么需要位运算？因为机器里的数字是表示方式和存储格式是二进制的。
十进制转二进制的方法
- 余数短除法除以二
- 降二次幂及减法混合运算
参考资料：[https://zh.wikihow.com/%E4%BB%8E%E5%8D%81%E8%BF%9B%E5%88%B6%E8%BD%AC%E6%8D%A2%E4%B8%BA%E4%BA%8C%E8%BF%9B%E5%88%B6][1]

##### 位运算符
- 左移（\<\<）： 0011 ——\> 0110
- 右移（\>\>）： 0110 ——\> 0011
- 按位或（|）： 0011|1011 ——\> 1011
- 按位与（&）： 0011&1011 ——\> 0011
- 按位取反（\~）： 0011 ——\> 1100
- 按位异或（^）： 0011^1011 ——\> 1000
##### 异或（XOR）
异或特点：相同为0，不同为1。
- x^0=x
- x^1s=\~x  //注意 1s=\~0
- x^(\~x)=1s
- x^x=0
- c=a^b =\> a^c=b, b^c=a //交换两个数 a^b^c=a^(b^c)=(a^b)^c //associative
##### 指定位置的位运算
- 将x最右边的n位清零：x&(\~0\<\<n)
- 获取x的第n位值(0或者1)：(x\>\>n)&1
- 获取x的第n位的幂值：x&(1\<\<n)
- 仅将第n位置为1：x|(1\<\<n)
- 仅将第n位置为0：x&(\~(1\<\<n))
- 将x最高位至第n位(含)清零：x&((1\<\<n)-1)
##### 实战位运算要点
- **判断奇偶**
	- x%2==1 —\> (x&1)==1 
	- x%2==0 —\> (x&1)==0
- **x\>\>1 —\> x/2** 
	- 即: x=x/2; —\> x=x\>\>1;
	- mid=(left+right)/2; —\> mid=(left+right)\>\>1;
- **X=X&(X-1) 清零最低位的1**
- **X&-X=\> 得到最低位的1**
- **X&\~X=\>0**

#### 布隆过滤器（Bloom Filter）
Bloom Filter是一个占用空间很小、效率很高的随机数据结构，它由一个bit数组和一组Hash算法构成。可用于判断一个元素是否在一个集合中，查询效率很高（1-N，最优能逼近于1）。
- 优点：空间效率和查询时间都远远超过一般算法。
- 缺点：有一定的误识别率和删除困难。

#### LRU Cache
**两个要素**
- 缓存大小
- 替换策略
	- LFU - least frequently used （频繁使用的最少者）
	- LRU - least recently used （最近使用的最少者）
**实现方式**
- hash table + double linked-list （哈希表+双向链表）
**时间复杂度**
- O(1)的查询、修改和更新

#### 排序算法
**比较类排序**：通过比较来决定元素间的相对次序，由于其时间复杂度不能突破
O(nlogn)，因此也称为非线性时间比较类排序。常见有：交换排序、插入排序、选择排序和归并排序。
**非比较类排序**：不通过比较来决定元素间的相对次序，它可以突破基于比较排序的时 间下界，以线性时间运行，因此也称为线性时间非比较类排序。常见有：计数排序、桶排序和基数排序。
##### 初级排序 - O(N^2)
1. 选择排序(Selection Sort) 
	每次找最小值，然后放到待排序数组的起始位置。
2. 插入排序(Insertion Sort) 
	从前到后逐步构建有序序列;对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
3. 冒泡排序(Bubble Sort) 
	嵌套循环，每次查看相邻的元素如果逆序，则交换。
##### 高级排序 - O(NlogN)
1. 快速排序(Quick Sort)
	数组取标杆**pivot**，将小元素放 pivot左边，大元素放右侧，然后依次对右边和右边的子数组继续快排；以达到整个序列有序。
```java
public static void quickSort(int[] array, int begin, int end) { 
	if (end <= begin) return;
	int pivot = partition(array, begin, end);
	quickSort(array, begin, pivot - 1);
    quickSort(array, pivot + 1, end);
}
static int partition(int[] a, int begin, int end) { 
	// pivot: 标杆位置，counter: 小于pivot的元素的个数 int pivot = end, counter = begin;
	for (int i = begin; i < end; i++) {
		if (a[i] < a[pivot]) {
			int temp = a[counter]; a[counter] = a[i]; a[i] = temp; counter++;
		} 
	}
	int temp = a[pivot]; 
	a[pivot] = a[counter]; 
	a[counter] = temp;
    return counter;
}
// 调用方式: quickSort(a, 0, a.length - 1)
```

2. 归并排序(Merge Sort) — 分治
	1. 把长度为n的输入序列分成两个长度为n/2的子序列。 
	2. 对这两个子序列分别采用归并排序。
	3. 将两个排序好的子序列合并成一个最终的排序序列。
```java
public static void mergeSort(int[] array, int left, int right) { 
	if (right <= left) return;
	int mid = (left + right) >> 1; // (left + right) / 2
    mergeSort(array, left, mid);
    mergeSort(array, mid + 1, right);
    merge(array, left, mid, right);
}

public static void merge(int[] arr, int left, int mid, int right) { 
	int[] temp = new int[right - left + 1]; // 中间数组
	int i = left, j = mid + 1, k = 0;
	while (i <= mid && j <= right) {
		temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
	}
	while (i <= mid) temp[k++] = arr[i++];
	while (j <= right) temp[k++] = arr[j++];
	for (int p = 0; p < temp.length; p++) { 
		arr[left + p] = temp[p];
	}
	// 也可以用 System.arraycopy(a, start1, b, start2, length) }
```

3. 归并和快排的特性
	1. 归并和快排具有相似性，但步骤顺序相反。
	2. 归并：先排序左右子数组，然后合并两个有序子数组。 
	3. 快排：先调配出左右子数组，然后对于左右子数组进行排序。

4. 堆排序(Heap Sort) — 堆插入 O(logN)，取最大/小值 O(1)
	1. 数组元素依次建立小顶堆
	2. 依次取堆顶元素，并删除
```cpp
void heap_sort(int a[], int len) {
    priority_queue<int,vector<int>,greater<int> > q;
	for(int i = 0; i < len; i++) { 
		q.push(a[i]);
	}
	for(int i = 0; i < len; i++) { 
		a[i] = q.pop();
	} 
}
static void heapify(int[] array, int length, int i) { 
	int left = 2 * i + 1, right = 2 * i + 2;
	int largest = i;
	if (left < length && array[left] > array[largest]){ 
		largest = leftChild;
    }
    if (right < length && array[right] > array[largest]) {
		largest = right; 
	}
	if (largest != i) {
		int temp = array[i]; 
		array[i] = array[largest]; 
		array[largest] = temp; 
		heapify(array, length, largest);
	} 
}
public static void heapSort(int[] array) { 
	if (array.length == 0) return;
	int length = array.length;
	for (int i = length / 2-1; i >= 0; i-){
        heapify(array, length, i);
	}
	for (int i = length - 1; i >= 0; i--) {
		int temp = array[0]; 
		array[0] = array[i]; 
		array[i] = temp; 
		heapify(array, i, 0);
	} 
}
```

##### 特殊排序 - O(N)
1. 计数排序(Counting Sort) 
	计数排序要求输入的数据必须是有确定范围的整数。将输入的数据值转化为键存储在额外开辟的数组空间中；然后依次把计数大于 1 的填充回原数组。
2. 桶排序(Bucket Sort)
	桶排序 (Bucket sort)的工作的原理：假设输入数据服从均匀分布，将数据分到有限数量的桶里，每个桶再分别排序(有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排)。
3. 基数排序(Radix Sort)
	基数排序是按照低位先排序，然后收集；再按照高位排序，然后再收集；依次类推，直到最高位。有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序。

### 本周leetcode练习总结
#### 剑指offer59 - I.滑动窗口最大值（本周复习|同239题）
1. 单调队列
	1. 算法思路
		1. 因为滑动窗口大小k是固定的，所以当滑动窗口从数组nums左侧进入时，可以分为**未形成窗口**和**形成窗口后**两个阶段。那么可以分别在两个阶段里循环判断窗口里的最大值。
		2. 未形成窗口阶段
			1. 数组nums的下标i在[0,k)之间递增循环。
			2. 当队列deque不为空，且当nums[i]\>deque[-1]时，删除队列中所有小于nums[i]的元素，保持队列中元素的单调递减；此时deque[0]为窗口中的最大值，将deque[0]加入结果res中。
		3. 形成窗口后阶段
			1. 数组nums的下标i在[k,len(nums))之间递增循环。
			2. 当队列首端元素deque[0]==nums[i-k]时，表示当前队列的最大元素已经移出了滑动窗口，所以要删除。
			3. 当队列deque不为空，且队列的末端元素deque[-1]\<nums[i]时，删除队列中所有小于nums[i]的元素，保持队列中元素的单调递减性。
			4. 当nums[i]不符合2、3点时，将nums[i]添加进队列deque；同时将队列首端元素deque[0]加入结果res中。
		4. 最后返回结果res即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，虽然分为两个阶段，但只遍历了一次数组nums。
		2. 空间复杂度：O(N)，队列deque最多存储k个元素，N=k。
	3. 题解代码
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        # 当nums为空，或k=0时
        if not nums or k == 0:
            return res
        # 初始化单调队列
        from collections import deque
        queue = deque()
        # 未形成窗口时
        for i in range(k):
            # 当queue不为空，且尾部元素<num[i]
            while queue and queue[-1] < nums[i]:
                queue.pop()
            queue.append(nums[i])
        res.append(queue[0])
        # 形成窗口后
        for i in range(k, len(nums)):
            # 当队列首部元素移出窗口后，删除
            if queue[0] == nums[i - k]:
                queue.popleft()
            # 当queue不为空，且尾部元素<num[i]
            while queue and queue[-1] < nums[i]:
                queue.pop()
            # 不符合上面2点的nums[i]加入到队列中
            queue.append(nums[i])
            # 将队列中的最大元素加入结果中
            res.append(queue[0])
        return res
```

2. 单调队列（简洁高效）
	1. 算法思路
		1. 设队列win用于保存数组nums的下标i；res保存结果。
		2. 遍历数组nums，下标为i，进行以下判断：
			1. 当i\>=k，且win[0]\<=i-k，即表示win[0]代表的元素下标已经移出了窗口大小范围，所以要从队列win中移除，win.pop(0)。
			2. 当队列win不为空，且队列末端代表的数组元素nums[win[-1]]\<nums[i]时，移出win的末端元素，win.pop()；保持队列win保存的下标在nums中所代表元素的单调性。
			3. 将不符合1、2点的下标i添加到win中。
			4. 当i\>=k-1时，即窗口成形开始，将元素nums[win[0]]加入到结果res中。
		3. 最终返回res即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，一次遍历数组，数组长度为N。
		2. 空间复杂度：O(N)，N=滑动窗口的大小k。
	3. 题解代码
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        win = []  # 保存nums下标i
        # 遍历数组nums
        for i in range(len(nums)):
            # 当win的首端元素不在滑动窗口k的范围内后，要删除
            if i >= k and win[0] <= i - k:
                win.pop(0)
            # 当win不为空，且win末端代表的元素小于i代表的元素时，需要删除
            while win and nums[win[-1]] < nums[i]:
                win.pop()
            # 不满足上述的i都加入win中
            win.append(i)
            # 当窗口形成后，开始将每次移动后的窗口最大值加入结果
            if i >= k - 1:
                res.append(nums[win[0]])
        # 返回结果
        return res
```

#### 24.两两交换链表中的节点（本周复习）
1. 递归法
	1. 算法思路
		1. 递归终止条件：**当head为空或只有head.next一个节点时**，返回head。
		2. 递归工作：
			1. 如果链表中至少有两个节点，在两两交换后，原链表头节点head指向新链表的第二个节点newhead.next；原链表第二个节点head.next指向新链表头节点newhead。
			2. 链表中的其余节点两两交换由递归工作实现。
			3. 在第2点完成后，更新节点之间的指针关系，即完成整个链表的两两交换。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历链表的所有节点。
		2. 空间复杂度：O(N)，n为链表的节点个数，实际取决于递归时系统调用的栈。
	3. 题解代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # 终止条件:head为空或仅有head一个节点
        if not head or not head.next:
            return head
        # 新头指向原头next
        newhead = head.next
        # 原头next指向通过递归实现
        head.next = self.swapPairs(newhead.next)
        # 新头next指向原头，完成交换
        newhead.next = head
        return newhead
```

#### 367.有效的完全平方数（本周复习）
1. 二分查找
	1. 算法思路
		1. **应用标准的二分查找代码模板**。
		2. 设定left=2，right=num/2；当left\<=right时，令x=(left+right)/2进行迭代，查找可能解：
			1. 当x\*x=num时，num是有效的完全平方数，返回true。
			2. 当x\*x\<num时，移动left=x+1
			3. 当x\*x\>num时，移动right=x-1
			4. 以上都不符合返回false。
		3. 考虑特例：当num\<2时，直接返回true。
	2. 复杂度分析
		1. 时间复杂度：O(logN)，二分查找时间复杂度
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        # num<2:true
        if num < 2:
            return True
        left, right = 2, num // 2
        while left <= right:
            x = (left + right) >> 1  # 位运算 等同于(left+right)//2
            if x * x == num:
                return True
            elif x * x < num:
                left = x + 1
            else:
                right = x - 1
        return False
```

2. 牛顿迭代法
	1. 算法思路
		1. 套用牛顿迭代法公式：**x=1/2\*(x+num/x)**
		2. 初始令x=num/2；当x\*x\>num时，通过1的公式迭代计算新的x，直到不满足循环条件。
		3. 最终返回x\*x==num的布尔值。
	2. 复杂度分析
		1. 时间复杂度：O(logN)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        x = num // 2
        while x * x > num:
            x = (x + num / x) // 2
        return x * x == num
```

#### 33.搜索旋转排序数组（本周复习）
1. 二分查找
	1. 算法思路
		1. **应用标准的二分查找代码模板，但需要对额外的场景做判断**。
		2. 设定left,right=0,n-1，n为nums的长度。
		3. 当left\<=right时，令mid=(left+right)/2，根据nums[mid]==target的情况进行判断：
			1. 当nums[mid]==target时，即找到目标，返回下标mid。
			2. 当nums[0]\<=nums[mid]，表示数组中0\~mid之间是有序的，此时进行额外判断target的情况：
				1. 当target落在nums[0]\~nums[mid]之间时，即满足nums[0]\<=target\<nums[mid]条件，令right=mid-1。
				2. 否则移动left=mid+1
			3. 当nums[0]\>nums[mid]时，表示数组中mid\~n-1之间是有序的，此时进行额外判断target的情况：
				1. 当target落在nums[mid]\~nums[n-1]之间时，即满足nums[mid]\<target\<=nums[n-1]条件，令left=mid+1。
				2. 否则移动right=mid-1
		4. 当left\<=right迭代完成，以上都不满足时，返回-1。
		5. 考虑，当数组nums为空时，直接返回-1。
	2. 复杂度分析
		1. 时间复杂度：O(logN)，二分查找的时间复杂度
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # nums为空，返回-1
        if not nums:
            return -1
        # 二分查找
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) >> 1  # mid=(left+right)/2
            if nums[mid] == target:  # 找到目标
                return mid
            elif nums[0] <= nums[mid]:  # 数组中0~mid是有序的
                # target落在nums[0]~nums[mid]之间，固定left，移动right
                if nums[0] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # 当nums[0]>nums[mid]
                # target落在nums[mid]~nums[n-1]之间，固定right，移动left
                if nums[mid] < target <= nums[n - 1]:
                    left = mid + 1
                else:
                    right = mid - 1
        # 找不到返回-1
        return -1
```

#### 191.位1的个数
1. 系统函数
	1. 算法思路
		1. 调用系统bin函数和count函数。
		2. 用count函数计算bin函数结果中1的个数。
	2. 复杂度分析
		1. 时间复杂度：O(1)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        return bin(n).count('1')
```

2. 十进制转二进制
	1. 算法思路
		1. n每次对2取余，判断余数是否为1；余数=1，则count+1。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次n的长度
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            res = n % 2
            if res == 1:
                count += 1
            n = n // 2
        return count
```

3. 位运算（重点）
	1. 算法思路
		1. 将n和1进行与运算：n&1，取得n的最低位，将其计入count。
		2. 再将n右移一位，n\>\>1，直到n迭代完毕。
	2. 复杂度分析
		1. 时间复杂度：O(1)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            # print(n & 1)
            count += n & 1
            n = n >> 1
        return count
```
4. 题解代码二
	1. 与题解代码一不同之处：位运算采用的是n&(n-1)的方式。
		2. n与n-1的与运算，总能把n的最低位的1变为0。
		3. 当n不为0时迭代，count计数+1，n&(n-1)逐渐将n中的1变为0后，迭代停止，返回count。
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n != 0:
            count += 1
            n = n & (n - 1)
            # print(n)
        return count
```

#### 198.打家劫舍（本周复习）
1. 动态规划
	1. 算法思路
		1. 数组nums的代表房间数量和每间房子可偷金额的大小。
		2. 如果nums\<=2，即偷两间房子里金额较大的那间即可。这里有两种情况：
			1. nums=1时，dp[0]=nums[0]
			2. nums=2时，dp[1]=max(nums[0],nums[1])
		3. 如果nums\>2，则做以下情况判断：
			1. 偷第i间房子，则i-1间房子不能偷窃，那么当前能偷到的最大金额为前i-2间房子的最高金额+第i间房子的金额：**dp[i]=dp[i-2]+nums[i]**。
			2. 不偷第i间房子，那么当前能偷到的最大金额为前i-1间房子的最大金额：**dp[i]=dp[i-1]**。
			3. 在1和2之间选择较大者，即得到前i间房子的最大金额。由此可得状态转移方程：**dp[i]=max(dp[i-2]+nums[i],dp[i-1])**。
		4. 结合2和3点，最终返回dp[n-1]，n为数组长度。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)，需要额外的数组空间保存状态。
	3. 题解代码
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # nums为空
        if not nums:
            return 0
        n = len(nums)
        # nums<2，返回较大者
        if n <= 2:
            return max(nums)
        # 动态规划
        # 初始化dp
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])  # 递推公式
        return dp[n - 1]
```
4. 空间压缩优化
	1. 因为前i间房子的最高金额只和前两间房子的最高金额相关，因此可以通过滚动数组，只保存前两间房子的最高金额。
		2. 设pre和cur起始均为0，可得递归公式：**cur,pre=max(pre+num,cur),cur**。
		3. 最终返回cur即可。
		4. 复杂度分析
			1. 时间复杂度：O(N)，遍历一次数组
			2. 空间复杂度：O(1)，常数变量
		5. 题解代码
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        cur = pre = 0
        for num in nums:
            cur, pre = max(pre + num, cur), cur
        return cur
```

#### 36.有效的数独（本周复习）
1. 一次迭代（遍历）
	1. 算法思路
		1. 一个9X9的数独，要保证其有效性，要达成以下三个条件：
			1. 行中没有重复数字
			2. 列中没有重复数字
			3. 3X3的子数独中没有重复数字
		2. 子数独block的下标idx=(行/3)\*3+(列/3)。
		3. 对行、列和子数独block分别建立哈希表来记录数字num出现的位置（某行或某列或某个block）和次数count。当某行或某列或某个block中的某个数字num出现的次数count\>1时，即出现了重复数字，数独是无效的。
		4. 当遍历完毕，没有出现上述重复情况，则数独是有效的。
	2. 复杂度分析
		1. 时间复杂度：O(1)，在固定的9X9的矩阵中一次遍历，复杂度不随数据变化而变化。
		2. 空间复杂度：O(1)，理由同时间复杂度。
	3. 题解代码
```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 初始化哈希表
        row = [{} for _ in range(9)]
        col = [{} for _ in range(9)]
        box = [{} for _ in range(9)]
        # 一次遍历9x9的矩阵
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j])
                    idx = (i // 3) * 3 + j // 3  # box下标
                    row[i][num] = row[i].get(num, 0) + 1
                    col[j][num] = col[j].get(num, 0) + 1
                    box[idx][num] = box[idx].get(num, 0) + 1
                    # 判断重复数字
                    if row[i][num] > 1 or col[j][num] > 1 or box[idx][num] > 1:
                        return False
        return True
```
4. 题解代码
	国际版参考，将哈希字典换成了set集合，不用记录次数。当数字在集合中已经有了，就表示出现重复了。
```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        
        rows = [set() for i in range(9)]
        cols = [set() for i in range(9)]
        mMat = [set() for i in range(9)]
        
        for i in range(9):
            for j in range(9):
                cur = board[i][j]
                if cur != '.':
                    
                    k = (i // 3 ) * 3 + j // 3
                
                    if cur not in rows[i]: rows[i].add(cur)
                    else: return False
                    
                    if cur not in cols[j]: cols[j].add(cur)
                    else: return False
                
                    if cur not in mMat[k]: mMat[k].add(cur)
                    else: return False
        return True
```

#### 190.颠倒二进制位
1. 逐位颠倒
	1. 算法思路
		1. 关键思想：对于位于索引**i**处的位，在反转之后，其位置应为**31-i**（索引从0开始）。
		2. 从右到左遍历输入整数的位字符串（即n=n\>\>1）；要检索最右边的位，应用与运算（n&1）。
		3. 对于每个位，将其反转到正确的位置（n&1\<\<31）后，将其添加到结果。
		4. 当n=0时，迭代终止。
	2. 复杂度分析
		1. 时间复杂度：O(logN)，循环迭代输入的非零整数位
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        power = 31
        while n:
            res += (n & 1) << power
            n = n >> 1
            power -= 1
        return res
```

2. 利用内置函数bin
	1. 算法思路
		1. **bin函数是返回一个整数型int或长整数long int的二进制表示**。
		2. 利用bin函数翻转输入的n，bin(n)[2:][:\:-1]
		3. 在其前面添加’0b’表示多个0；在其后面加上替代’0b’的多个0，‘0’\*(32-len(bin(n)[2:]))。
		4. 最终返回结果int的二进制表示。
	2. 复杂度分析
		1. 时间复杂度：O(1)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:                                                                
    def reverseBits(self, n: int) -> int:                                      
        return int('0b' + bin(n)[2:][::-1] + '0' * (32 - len(bin(n)[2:])), 2)  
```
4. 题解代码二
	填充0使用了内置**zfill**函数。
```python
class Solution:                                    
    def reverseBits(self, n: int) -> int:          
        return int(bin(n)[2::].zfill(32)[::-1], 2) 
```

#### 42.接雨水（本周复习）
1. 暴力法
	1. 算法思路
		1. 对于每根柱子i，找到位于其左边的最高柱子left\_max和其右边的最高柱子right\_max，则第i根柱子能接到的雨水量位min(left\_max,right\_max)-height[i]。
		2. 遍历所有柱子，计算每根柱子能接的雨水量，返回累加即可。
	2. 复杂度分析
		1. 时间复杂度：O(N^2)，遍历一次数组nums；left\_max和right\_max每次通过max函数更新，也是遍历了数组。
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        res = 0
        for i in range(1, len(height) - 1):
            # 找0～i的最高柱子
            left_max = max(height[:i + 1])
            # 找i~n-1的最高柱子
            right_max = max(height[i:])
            # 最高中的较小者-柱子高=雨水量
            res += min(left_max, right_max) - height[i]
        return res
```

2. 双指针
	1. 算法思路
		1. 设置left和right两个指针，分别从区间的左端、右端开始向中间靠拢。同时设置leftmax和rightmax分别记录左右移动过程中左边和右边最高柱子的高度。
		2. 当leftmax\<rightmax时，拿leftmax-height[left]得到雨水量，之后移动left。
		3. 反之，拿rightmax-height[right]得到雨水量，之后移动right。
		4. 最终返回累加雨水量即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        res = 0
        # 初始化
        left = 0
        right = len(height) - 1
        leftmax = height[0]
        rightmax = height[-1]
        # 一次遍历，双指针左右夹逼
        while left <= right:
            # 更新左右最大值
            leftmax = max(height[left], leftmax)
            rightmax = max(height[right], rightmax)
            # 取左右最大值中较小者，计算雨水量
            if leftmax < rightmax:
                res += leftmax - height[left]
                left += 1
            else:
                res += rightmax - height[right]
                right -= 1
        return res
```

3. 单调栈解法
	1. 算法思路
		1. 核心思想：**对于区间内任意位置的柱子，找其两侧最近的比其高的柱子**。
		2. 通过**单调栈**来解决最近值的问题。单调栈就是在栈**后进先出**的特性上，维持栈内元素的单调性。找某侧最近一个比其大的值，维持栈内元素递减；相反，找某侧最近一个比其小的值，维持栈内元素递增。
		3. 当某个位置的元素a出栈的时候，自然得到a位置左右两侧比其高的柱子：
			1. 一个是导致a位置元素出栈的柱子（a右侧柱子比a高）
			2. 一个是a出栈后的栈顶元素（a左侧柱子比a高）
		4. 当有了a位置左右两侧比a高的柱子后，便可以计算a位置可接的雨水量。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)，栈存储n个元素的空间
	3. 题解代码
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        # 单调栈
        stack = []
        res = 0
        for i in range(len(height)-1):
            # 栈不为空，且height[栈顶]<height[i]
            while len(stack) > 0 and height[stack[-1]] < height[i]:
                top = stack.pop()
                if len(stack) == 0:
                    break
                # 高度
                # print(height[stack[-1]],height[i],height[top])
                h = min(height[stack[-1]], height[i]) - height[top]
                # 宽度
                w = i - stack[-1] - 1
                # 雨水量
                res += (w * h)
            # i入栈
            stack.append(i)
        return res
```

#### 24.两两交换链表中的节点（本周复习）
1. 递归法
	1. 算法思路
		1. 递归终止条件：**当head为空或没有head.next时**，返回head。
		2. 递归工作：
			1. 如果链表中至少有两个节点，在两两交换后，原链表头节点head指向新链表的第二个节点newhead.next；原链表第二个节点head.next指向新链表头节点newhead。
			2. 链表中的其余节点两两交换由递归工作实现。
			3. 在第2点完成后，更新节点之间的指针关系，即完成整个链表的两两交换。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历链表的所有节点。
		2. 空间复杂度：O(N)，n为链表的节点个数，实际取决于递归时系统调用的栈。
	3. 题解代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        newhead=head.next
        head.next = self.swapPairs(newhead.next)
        newhead.next=head
        return newhead
```

#### 45.跳跃游戏II（本周复习）
1. 贪心算法
	1. 算法思路
		1. 设置cur当前跳跃的最大范围；nex下一跳的最远距离；step跳跃步数。
		2. 当数组nums的下标i在cur的范围内，则更新nex的最远距离，**最远距离=当前下标i+当前元素nums[i]的和**。
		3. 当下标i超出最远距离nex时，更新cur为nex，同时跳一步step+1。因为超出可跳的最远距离了，所以要跳一步。
		4. 当可以跳的最远距离可以达到或超出数组nums的长度时，即表示再跳一步就可以达到终点了，所以返回step+1即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        # nums只有1位时，不用跳跃
        if n == 1:
            return 0
        # 设置可跳跃范围cur，下次可跳最远距离nex，步数step
        cur = nex = step = 0
        # 遍历一次数组
        for i in range(n):
            # 当位于的i超出可跳范围时
            if i > cur:
                cur = nex
                step += 1
            nex = max(nex, i + nums[i])  # 更新可以跳最远距离
            # 下一跳最远距离可以达到或超出终点
            if nex >= n - 1:
                return step + 1
```

2. 动态规划
	1. 算法思路
		1. 设置记录步数的数组dp，长度为n+1，n为数组nums的长度；**初始dp数组=0**。设定一个边界下标j=0。
		2. 遍历数组nums，可跳距离s为i+nums[i]，当s\>j时，将dp[j+1:s+1]，即跳跃覆盖范围(s-j)都更新为dp[i]+1。即得到状态转移方程：**dp[j+1:s+1]=[dp[i]+1]\*(s-j)**。
		3. 当s\>n-1时，表示已跳跃到终点，终止遍历；返回dp[-1]即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)，dp数组需要额外空间
	3. 题解代码
```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        dp = [0] * len(nums)
        k = 0
        for i, n in enumerate(nums):
            if i + n > k:
                dp[k + 1:i + n + 1] = [dp[i] + 1] * (i + n - k)
                # print(dp)
                k = i + n
                if i + n >= len(nums) - 1:
                    break
        # print(dp)
        return dp[-1]
```

#### 1122.数组的相对排序
1. 计数排序
	1. 算法思路
		1. 利用额外数组cnt保存arr1数组中每个元素num和其次数的对应关系。
		2. 当元素num在arr2中时，在结果res中扩展相应的num和次数cnt[num]；使用过的num，将其次数在cnt中置为0。
		3. 再遍历一次cnt数组，将不在arr2中出现的元素和其次数，依次加入到结果res中，最终返回res即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，N=arr1长度m+arr2长度n+cnt长度cnt。
		2. 空间复杂度：O(N)，N为cnt数组空间。
	3. 题解代码
```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        res = []
        # 建立额外数组cnt，记录arri的元素和次数
        cnt = [0] * (max(arr1) + 1)
        for num in arr1:
            cnt[num] += 1
        # print(cnt)
        # 遍历arr2，将arr2中出现的元素加入res
        for num in arr2:
            res.extend([num] * cnt[num])
            cnt[num] = 0  # 使用过的元素次数置为0
        # print(res)
        # print(cnt)
        # 遍历不在arr2出现的元素
        for i in range(len(cnt)):
            if cnt[i] > 0:
                res.extend([i] * cnt[i])
        return res
```

#### 46.全排列（本周复习）
1. 回溯法
	1. 算法思路
		1. 利用DFS遍历树的所有节点，对所有节点增加使用状态标识。向下递归的时候，节点打上“使用”标识；向上回溯的时候，对已使用的节点打上“撤销”标识，以便在后续组合中继续使用。
		2. 递归工作：
			1. 建立used数组，用于保存所有节点的使用状态，初始为false，当使用过后标为true。
			2. 终止条件：设定向下层次变量depth，当depth==nums长度时，表示排列的长度已经足够，可以加入到结果res中。
			3. 对未使用过的节点进行进行以下操作
				1. 将节点的used状态标为true，表示已使用。
				2. 将节点加入中间结果tmp中
				3. 继续递归工作，递归下一层时depth+1，同时也将中间结果tmp和状态used一并传入。
			4. 递归完成后进行回溯，将使用过的节点的状态标为false，以便后续使用；同时从中间结果tmp中删除已使用的节点。
	2. 复杂度分析：
		1. 时间复杂度：O(N\*N!)，遍历数组为N，每个内部节点循环为N，非叶子节点时间为O(N\*N!)；最后一层有N!个叶子节点，结果拷贝需要O(N)，时间复杂度同为O(N\*N!)。最终时间复杂度O(N\*N!)。
		2. 空间复杂度：O(N\*N!)，递归深度logN，全排列个数N！，每个排列为N，最终为O(N\*N!)。
	3. 题解代码
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        # uesd数组，记录节点状态
        used = [False] * n
        res = []

        # 回溯
        def dfs(depth, tmp, used):
            # 递归终止：当depth==n时，表示排列完成
            if depth == n:
                res.append(tmp.copy())
                return
                # 遍历节点
            for i in range(n):
                # 未使用的节点处理
                if not used[i]:
                    used[i] = True
                    tmp.append(nums[i])
                    # 递归工作
                    dfs(depth + 1, tmp, used)
                    # 状态重置
                    used[i] = False
                    tmp.pop()

        # 从第一层开始递归
        dfs(0, [], used)
        return res
```
4. 动态数组优化空间
	1. 算法思路
		1. 设定递归层数depth，在depth～n之间迭代下标i，交换nums[depth]与nums[i]的位置。
			2. 递归终止条件：当depth==n时，表示新排列产生完成，将其加入结果。
			3. 递归工作完成时，将nums[depth]与nums[i]的位置恢复，以便后续使用。
		2. 复杂度分析
			1. 时间复杂度：O(N\*N!)
			2. 空间复杂度：O(N)，递归深度为N。
		3. 题解代码
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []

        # 回溯
        def dfs(depth):
            # 终止条件
            if depth == n:
                res.append(nums[:])
                return
            # 处理当前节点
            for i in range(depth, n):
                nums[depth], nums[i] = nums[i], nums[depth]
                dfs(depth + 1)
                # 状态重置
                nums[depth], nums[i] = nums[i], nums[depth]

        # 递归
        dfs(0)
        return res
```

2. 分治法
	1. 算法思路
		1. 锚定nums[i]，对nums数组中的剩余元素nums[:i]和nums[i+1:]进行排列组合。
		2. nums[:i]和nums[i+1:]的排列组合，调用permute函数自身进行递归工作；递归终止条件为传入permute函数的nums为空，返回[]。
		3. 递归后将nums[i]与分治结果加入最终结果res中。
	2. 复杂度分析：
		1. 时间复杂度：O(N\*N!)
		2. 空间复杂度：O(N\*N!)
	3. 题解代码
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # 终止条件
        if not nums:
            return [[]]
        # 遍历nums
        for i in range(len(nums)):
            subres = self.permute(nums[:i] + nums[i + 1:])
            for p in subres:
                res.append([nums[i]] + p)
        return res
```

#### 231.2的幂
1. 朴素算法
	1. 算法思路
		1. 利用数学特性：当n%2=0时不断迭代，同时n=n/2，最终返回n==1的布尔值。
	2. 复杂度分析
		1. 时间复杂度：O(logN)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        while n % 2 == 0:
            n = n // 2
        return n == 1
```

2. 位运算
	1. 算法思路（获取最右边的1）
		1. 获取二进制中最右边的1。通过x&(-x)得到最低位的1，其他位设为0。
		2. 若x为2的幂，则它的二进制表示中只包含一个1，则有x&(-x)=x。
		3. 返回x&(-x)==x的结果即可。
	2. 复杂度分析
		1. 时间复杂度：O(1)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        return n & (-n) == n
```
4. 另一个算法思路（去除二进制最右边的1）
	1. 若n=2的幂次方，则一定满足以下条件：
		1. 恒有n&(n-1)==0，因为：
			1. n的二进制最高位为1，其余为0
				2. n-1的二进制最高位为0，其余为1
			2. 满足n\>0。
		2. 通过n\>0，且n&(n-1)==0的结果判断。
		3. 复杂度分析
			1. 时间复杂度：O(1)
			2. 空间复杂度：O(1)
		4. 题解代码
```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        return n & (n - 1) == 0
```

#### 338.比特位计数
1. 位运算
	1. 算法思路
		1. 迭代统计每个数中二进制为1的个数。
		2. 通过n&(n-1)消掉最低位1的方法，累加次数直到n为0。
	2. 复杂度分析
		1. 时间复杂度：O(N\*M)，N是数字从0～n的迭代次数，M是每个数字的二进制迭代长度。
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        # 计数数字的二进制中1的次数
        def bin_count(n):
            count = 0
            while n:
                n = n & (n - 1)
                count += 1
            return count

        res = []
        # 迭代0～n
        for i in range(num + 1):
            res.append(bin_count(i))
        return res
```

#### 718.最长重复子数组
1. 动态规划
	1. 算法思路
		1. 设定i是numsA的下标，j是numsB的下标。通过双层循环找出所有i和j的组合，则有
			1. 当A[i]==B[j]时，有dp[i][j]=dp[i-1][j-1]+1。
			2. 当A[i]!=B[j]时，dp[i][j]=0。
		2. 循环过程记录状态值，最终返回最大值即可。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，m和n分别为numsA和numsB的长度
		2. 空间复杂度：O(M\*N)，dp数组是二维数组，矩阵为m\*n。
	3. 题解代码
```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        # 初始化
        m, n = len(A), len(B)
        res = 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # 遍历数组
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    res = max(res, dp[i][j])
        # print(dp)
        return res
```
4. 测试用例参考
```python
测试用例:[1,2,3,2,1]
	    [3,2,1,4,7]
测试结果:3
期望结果:3
stdout:
[[0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 1, 0, 0], 
 [0, 0, 1, 0, 0, 0], 
 [0, 1, 0, 0, 0, 0], 
 [0, 0, 2, 0, 0, 0], 
 [0, 0, 0, 3, 0, 0]]
```

2. 滑动窗口
	1. 算法思路
		1. 将A和B固定对齐后进行比较。
		2. 假设B的长度为n，A的长度为m；如果固定A，依次迭代B，生成滑动窗口的大小为length=min(m,n-i)。
		3. 窗口内遍历过程
			1. 依次遍历窗口内重叠部分，当有A[i]==B[i]时，则k+1，否则k置为0；如此连续如果有连续相等的子数组时，k会连续加一。
			2. 最终返回最大的k即可。
		4. 先固定A，通过滑动窗口遍历B，再固定B，通过滑动窗口遍历A；最终返回两者之间最大的连续子数组长度即可。
	2. 复杂度分析
		1. 时间复杂度：O(M\+N)\*min(M,N)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        # 定义滑动窗口
        def maxLength(addA, addB, length):
            ret = k = 0
            for i in range(length):
                if A[addA + i] == B[addB + i]:
                    # print(i,A[addA+i],B[addB+i])
                    k += 1
                    ret = max(ret, k)
                else:
                    k = 0
            return ret

        m, n = len(A), len(B)
        ret = 0
        for i in range(n):
            #滑动窗口大小
            length = min(m, n - i)
            # print(i,length,maxLength(i,0,length))
            #固定B，滑动A
            ret = max(ret, maxLength(i, 0, length))
        for i in range(m):
            length = min(n, m - i)
            # print(i, length, maxLength(i, 0, length))
            ret = max(ret, maxLength(0, i, length))
        return ret
```

#### 47.全排列II（本周复习）
1. 回溯法
	1. 算法思路
		1. 因为给定的数组包含重复数字，要求返回不重复的全排列。故在全排列的算法基础上，增加了剪枝操作，删除重复的排列组合。
		2. 为后续剪枝操作的方便，首先需要对数组做排序处理。
		3. 同时满足以下条件时，需要做剪枝操作，即跳过当前元素。
			1. 当前元素不是第一个元素，即下标i\>0。
			2. 当前元素与上一个元素相同，即nums[i]==nums[i-1]
			3. 上一个元素已被使用，即used[i-1]=True
	2. 复杂度分析
		1. 时间复杂度：O(N\*N!)
		2. 空间复杂度：O(N\*N!)
	3. 题解代码
```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        # uesd数组，记录节点状态
        used = [False] * n
        res = []

        # 回溯
        def dfs(depth, tmp, used):
            # 递归终止：当depth==n时，表示排列完成
            if depth == n:
                res.append(tmp.copy())
                return
                # 遍历节点
            for i in range(n):
                # 未使用的节点处理
                if not used[i]:
                    # 剪枝
                    if i > 0 and nums[i] == nums[i - 1] and used[i - 1] is True:
                        continue
                    # 正常处理
                    used[i] = True
                    tmp.append(nums[i])
                    # 递归工作
                    dfs(depth + 1, tmp, used)
                    # 状态重置
                    used[i] = False
                    tmp.pop()

        # 从第一层开始递归
        dfs(0, [], used)
        return res
```


[1]:	https://zh.wikihow.com/%E4%BB%8E%E5%8D%81%E8%BF%9B%E5%88%B6%E8%BD%AC%E6%8D%A2%E4%B8%BA%E4%BA%8C%E8%BF%9B%E5%88%B6