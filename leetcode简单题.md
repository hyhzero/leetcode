## Leetcode简单

[TOC]

### 1. 整数反转

**方法1** 整数转字符串，字符串反转，字符串转整数，再判断范围

```python
def reverse(self, x: int) -> int:
    sign = 1 if x >= 0 else -1
    x = int(str(abs(x))[::-1])
    x = sign*x
    if x >= (-2**31) and x <= (2**31-1):
        return x
    else:
        return 0
```

```python
def reverse(self, x):
    sign = 1 if x > 0 else -1
    x = str(abs(x))
    x = list(x)
    x = list(reversed(x))
    x = int(''.join(x))
    if (-2**31) <= x and x < 2**31:
        return sign*int(x)
    else:
        return 0
```

**方法2 溢出判断**

<img src="C:\Users\zero\AppData\Roaming\Typora\typora-user-images\image-20200125124723265.png" alt="image-20200125124723265" style="zoom: 67%;" />

注意：在取模操作上，C/C++/Java是向0取整，python是向下取整

在python中 ：-53除以10=-6 …7 所以python中 -53%10=7
在c语言中，-53除以10=-5 … -3 所以c语言中 -53%10=-3

**python整除//**

-123/10 = -12.3，向下取整得-13，所以-123//10=-13

```python
def reverse(self, x: int) -> int:
    INTMAX = 2**31

    rev = 0
    sign = 1 if x >= 0 else -1
    x = abs(x)
    while x!=0:
        pop = x%10

        if (rev > (INTMAX//10) or rev==(INTMAX//10) and pop > 7):
            return 0

        rev = rev*10 + pop
        x = x//10

        return int(sign*rev)
```

自定义模运算

```python
import math
class Solution:

    def div(self, a, b):
        d = 1.0*a/b
        return math.floor(d) if d >= 0 else math.ceil(d)

    def mod(self, a, b):
        return a - self.div(a, b)*b

    def reverse(self, x: int) -> int:
        INTMAX = 2**31-1
        INTMIN = -2**31
        

        rev = 0
        sign = 1 if x > 0 else -1
        while x!=0:
            pop = self.mod(x, 10)

            if sign==1 and rev > self.div(INTMAX, 10) or rev==self.div(INTMAX
            , 10) and pop > self.mod(INTMAX, 10):
                return 0
            if sign==-1 and rev < self.div(INTMIN, 10) or rev==self.div(INTMIN, 10) and pop < self.mod(INTMIN, 10):
                return 0
            
            rev = rev*10 + pop
            x = self.div(x, 10)

        return rev
```

### 2. 两数之和

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

![image-20200125190051939](C:\Users\zero\AppData\Roaming\Typora\typora-user-images\image-20200125190051939.png)

* **暴力解法**

  ```python
  def twoSum(self, nums, target):
      for i in range(len(nums)):
          for j in range(i, len(nums)):
              if nums[i] + nums[j]==target and i!=j:
                  return i, j
  ```

* 两次遍历

  ```python
  def twoSum(self, nums, target):
  	hashmap = {}
      for index, x in enumerate(nums):
          hashmap[x] = index
      for index, x in enumerate(nums):
          if target-x in hashmap.keys() and index!=hashmap[target-x]:
              return [hashmap[target-x], index]
      return None
  ```

* 一次遍历

  ```python
  def twoSum(self, nums, target):
      hashmap = {}
      for index, x in enumerate(nums):
          if target-x in hashmap.keys():
              return [hashmap[target-x], index]
         	hashmap[x] = index
          
      return None
  ```

  

### 3. 回文数

* 整数转字符串

  ```python
  def isPalindrome(self, x):
      if str(x)[::-1]==str(x):
          return True
      else:
          return False
  ```

* 反转

  ```python
  def isPalindrome(self, x: int) -> bool:
      if x < 0:
          return False
  
      rev = 0
      tmp = x
      while x!=0:
          rev = rev*10 + x%10
          x = x//10
          return rev==tmp
  ```

* 反转一半

  ```python
  def isPalindrome(self, x: int) -> bool:
      if x < 0 or (x%10==0 and x!=0):
          return False
  
      rev = 0
      while rev < x:
          rev = rev*10 + x%10
          x = x//10
  
      return x==rev or x==(rev//10)
  ```

  x末尾为0，不容易判断反转一半

### 4. 罗马数字转整数

* 字符映射

  ```python
  class Solution:
      def romanToInt(self, s: str) -> int:
          roman = ['I', 'V', 'X', 'L', 'C', 'D', 'M', 'IV', 'IX', 'XL', 'XC', 'CD', 'CM']
          num = [1, 5, 10, 50, 100, 500, 1000, 4, 9, 40, 90, 400, 900]
          rn = dict(zip(roman, num))
  
          sums = 0
          i = 0
          while i < len(s):
              if (i+1) < len(s) and s[i: i+2] in rn:
                  k = s[i: i+2]
                  sums = sums + rn[k]
                  i = i + 2
              else:
                  k = s[i]
                  sums = sums + rn[k]
                  i = i + 1
  
          return sums
  ```

* 逐个翻译

  ```python
  class Solution:
      def romanToInt(self, s: str) -> int:
          roman = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
          num = [1, 5, 10, 50, 100, 500, 1000]
          rn = dict(zip(roman, num))
          trans = []
          
          for x in s:
              trans.append(rn[x])
          for i in range(len(trans)):
              if i < len(trans)-1 and trans[i] < trans[i+1]:
                  trans[i] = trans[i]*(-1)
          
          return sum(trans)
  ```

  

* 更快翻译

  ```python
  class Solution:
      def romanToInt(self, s: str) -> int:
          roman = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
          num = [1, 5, 10, 50, 100, 500, 1000]
          rn = dict(zip(roman, num))
  		
          sums = 0
          prev = 0
          for x in reversed(s):
              v = rn[x]
              if v < prev:
                  sums += -v
              else:
                  sums += v
              prev = v
          return sums
  ```

### 5. 有效的括号

* 栈

  ```python
  class Solution:
      def isValid(self, s: str) -> bool:
  
          stack = []
  
          for x in s:
              if len(stack)==0:
                  stack.append(x)
              elif x==']' and stack[-1]=='[' or x==')' and \
              stack[-1]=='(' or x=='}' and stack[-1]=='{':
                  stack.pop()
              else:
                  stack.append(x)
          
          return len(stack)==0
  ```

  

### 6. 合并两个有序链

* 哨兵节点

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
          head = ListNode(0)
          p = head
          while l1 and l2:
              if l1.val < l2.val:
                  p.next = l1
                  l1 = l1.next
              else:
                  p.next = l2
                  l2 = l2.next
              p = p.next
  
          if l1:
              p.next = l1
          else:
              p.next = l2
  
          return head.next
  ```

* 递归

  ```python
  class Solution:
      def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
          if l1==None:
              return l2
          if l2==None:
              return l1
  
          if l1.val < l2.val:
              l1.next = self.mergeTwoLists(l1.next, l2)
              return l1
          else:
              l2.next = self.mergeTwoLists(l1, l2.next)
              return l2
  ```

### 7. 删除排序数组中的重复项

**双指针**

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:

        i = 0
        for j in range(len(nums)):
            if nums[i]!=nums[j]:
                nums[i+1] = nums[j]
                i = i + 1

        return i + 1
```

### 8. 移除元素

* 双指针

  ```python
  class Solution:
      def removeElement(self, nums: List[int], val: int) -> int:
  
          i = 0
          for j in range(len(nums)):
              if nums[j]!=val:
                  nums[i]=nums[j]
                  i = i + 1
  
          return i
  ```

### 9. 搜索插入位置

* 二分查找

  ```python
  class Solution:
      def searchInsert(self, nums: List[int], target: int) -> int:
  
          i = 0
          j = len(nums)-1
  
          mid = (i + j)//2
  
          while i <= j:
              if nums[mid]==target:
                  return mid
              elif nums[mid] < target:
                  i = mid + 1
              else:
                  j = mid - 1
  
              mid = (i+j)//2
  
          return i
  ```


### 10. 最长公共前缀

* 普通解法

  ```python
  class Solution:
      def longestCommonPrefix(self, strs: List[str]) -> str:
          if len(strs)==0:
              return ""
              
          pattern = strs[0]
          lens = [len(s) for s in strs]
          min_len = min(lens)
  
  
          i = 0
          while i < min_len:
              p = pattern[i]
  
              stop = False
  
              for s in strs:
                  if s[i]!=p:
                      stop=True
                      break
  
              if stop==True:
                  break
  
              i = i + 1
  
          if i==0:
              return ""
          else:
              return pattern[:i]
  ```

* 水平扫描法

  ```python
  # 函数
  class Solution:
      def common_preifx(self, pattern, s):
          if pattern=='' or s=='' or pattern[0]!=s[0]:
              return ''
          i = 0
          while i < len(pattern) and i < len(s) and pattern[i]==s[i]:
              i = i + 1
          return pattern[:i]
  
      def longestCommonPrefix(self, strs: List[str]) -> str:
          if len(strs)==0:
              return ''
  
          pattern = strs[0]
  
          for s in strs:
              pattern = self.common_preifx(pattern, s)
          
          return pattern
      
  # 非函数
  class Solution:
      def longestCommonPrefix(self, strs: List[str]) -> str:
          if len(strs)==0:
              return ''
  
          pattern = strs[0]
  
          for s in strs:
              if pattern=='' or s=='' or pattern[0]!=s[0]:
                  return ''
  
              i = 0
              while i < len(pattern) and i < len(s) and pattern[i]==s[i]:
                  i = i + 1
              pattern = pattern[:i]
          
          return pattern
  ```

### 11. 两数之和II - 输入有序数组

* 一次遍历

  ```python
  class Solution:
      def twoSum(self, numbers: List[int], target: int) -> List[int]:
  
          hashmap = {}
  
          for index, num in enumerate(numbers):
              if (target - num) in hashmap:
                  return [hashmap[target-num], index+1]
              
              hashmap[num] = index+1
          
          return None
  ```

* 双指针法（使用排序性质）

  ```python
  class Solution:
      def twoSum(self, numbers: List[int], target: int) -> List[int]:
  
          i = 0
          j = len(numbers)-1
          while i < j:
              if numbers[i] + numbers[j] == target:
                  return [i+1, j+1]
              elif numbers[i] + numbers[j] < target:
                  i = i + 1
              else:
                  j = j - 1
          return None
  ```

### 12. 删除排序链表中的重复元素

* 方法同7

  ```python
  class Solution:
      def deleteDuplicates(self, head: ListNode) -> ListNode:
  
          p = head
          q = head
          while q:
              if q.val!=p.val:
                  p.next = q
                  p = p.next
              
              if q.next==None and q.val==p.val:
                  p.next=q.next
                  break
                  
              q = q.next
          
  
          return head
  ```

* 直接法

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def deleteDuplicates(self, head: ListNode) -> ListNode:
  
          if head==None:
              return None
  
          p = head
          q = head.next
          while q:
              if p.val!=q.val:
                  p = q
                  q = q.next
              else:
                  p.next = q.next
                  q = q.next
  
          return head
  ```

### 13. 旋转数组

* 暴力解法(超时)

  ```python
  class Solution:
      def rotate(self, nums: List[int], k: int) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
  
          # 暴力
          for _ in range(k):
              p = nums[-1]
              for i in range(len(nums)):
                  tmp = nums[i]
                  nums[i] = p
                  p = tmp
  ```

* 环状替换(计数)

  ```python
  class Solution:
      def rotate(self, nums: List[int], k: int) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
          k %= len(nums)
          if k==0:
              return
          
          count = 0
          start = 0
          while count < len(nums):
              prev = nums[start]
              current = (start + k)%len(nums)
              while start!=current:
                  temp = nums[current]
                  nums[current] = prev
                  prev = temp
                  current = (current + k)%len(nums)
                  count += 1
              nums[start] = prev
              count += 1
              start += 1
  ```
  
  
  
* 三次反转

  ```python
  class Solution:
      def rotate(self, nums: List[int], k: int) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
          k = k%len(nums)
          
          for i in range(len(nums)//2):
              tmp = nums[i]
              nums[i] = nums[len(nums)-i-1]
              nums[len(nums)-i-1] = tmp
  
          for i in range(k//2):
              tmp = nums[i]
              nums[i] = nums[k-i-1]
              nums[k-i-1] = tmp
          
          i = k
          j = len(nums)-1
          while i < j:
              tmp = nums[i]
              nums[i] = nums[j]
              nums[j] = tmp
              i = i + 1
              j = j - 1
  ```

### 14. 第一个错误的版本

* 二分法

  ```python
  # The isBadVersion API is already defined for you.
  # @param version, an integer
  # @return a bool
  # def isBadVersion(version):
  
  class Solution:
      def firstBadVersion(self, n):
          """
          :type n: int
          :rtype: int
          """
          if n==1:
              return 1
  
          left = 1
          right = n
          mid = (left + right)//2
          while left <= right:
              if isBadVersion(mid)==True:
                  if isBadVersion(mid-1)==False:
                      return mid
                  else:
                      right = mid - 1
              else:
                  left = mid + 1
  
              mid = (left + right)//2
  
          return -1
  ```

### 15. 有效完全平方数

* 平方数的性质

  数学定理(1 + 3 + 5 + ... + (2n - 1) = n ^ 2)
  
  ```python
  class Solution:
      def isPerfectSquare(self, num: int) -> bool:
  
          i = 1
          while num > 0:
              num -= i
              i += 2
  
          return num==0
  ```
  
  
  
* 二分法

  ```python
  class Solution:
      def isPerfectSquare(self, num: int) -> bool:
  
          left = 1
          right = num
          mid = (left + right)//2
          while left <= right:
              if mid**2==num:
                  return True
              elif mid**2 < num:
                  left = mid + 1
              else:
                  right = mid - 1
              mid = (left + right)//2 
          return False
  ```

### 16. 完美数

* 枚举

  ```python
  class Solution:
      def checkPerfectNumber(self, num: int) -> bool:
          if num <= 2:
              return False
  
          sums = 1
          i = 2
          while i*i <= num:
              if num%i==0:
                  sums = sums + i
                  if i*i!=num:
                      sums = sums + (num/i)
  
              i = i + 1
  
          return sums==num
  ```

### 17. 平方数之和

* 二分法

  ```python
  import math
  class Solution:
      def judgeSquareSum(self, c: int) -> bool:
          
          right = int(math.sqrt(c)) + 1
          left = 0
          while left <= right:
              if left**2 + right**2 == c:
                  return True
              
              if left**2 + right**2 < c:
                  left = left + 1
              else:
                  right = right - 1
  
          return False
  ```

  

### 18. 子数组的最大平均数

* 累计求和

  ```python
  class Solution:
      def findMaxAverage(self, nums: List[int], k: int) -> float:
  
          sums = []
  
          prev = sum(nums[:k])
          sums.append(prev)
          for i in range(k, len(nums)):
              current = prev + nums[i] - nums[i-k]
              sums.append(current)
              prev = current
  
          return max(sums)/k
  ```

* 滑动窗口

  ```python
  class Solution:
      def findMaxAverage(self, nums: List[int], k: int) -> float:
  
          max_sum = sum(nums[:k])
          prev = max_sum
  
          for i in range(k, len(nums)):
              current = prev + nums[i] - nums[i-k]
              if max_sum < current:
                  max_sum = current
              prev = current
  
          return max_sum/k
  ```

### 19. 三个数最大乘积

* 排序

  ```python
  class Solution:
      def maximumProduct(self, nums: List[int]) -> int:
  
          nums.sort()
  
          if nums[0] >= 0:
              return nums[-1]*nums[-2]*nums[-3]
          else:
              return max(nums[0]*nums[1]*nums[-1], nums[-1]*nums[-2]*nums[-3])
  ```

* 线性扫描

### 20. 分糖果

* 集合

  ```python
  class Solution:
      def distributeCandies(self, candies: List[int]) -> int:
          sister = set(candies)
  
          if len(sister) >= (len(candies)//2):
              return (len(candies)//2)
          else:
              return len(sister)
  ```

* 排序

  ```python
  class Solution:
      def distributeCandies(self, candies: List[int]) -> int:
          # 排序
          candies.sort()
  
          count = 1
          i = 0
          for j in range(len(candies)):
              if candies[i]!=candies[j]:
                  count = count + 1
                  i = j
                  
          if count > len(candies)//2:
              return len(candies)//2
          else:
              return count
  ```

### 21. 范围求和II

* 短板效应

  ```python
  import numpy as np
  class Solution:
      def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
  
          for (r, c) in ops:
              m = min(m, r)
              n = min(n, c)
  
          return m*n
  ```


### 22. 最长和谐子序列

* 计数法

  ```python
  class Solution:
      def findLHS(self, nums: List[int]) -> int:
          counts = {}
  
          for num in nums:
              if num in counts:
                  counts[num] = counts[num] + 1
              else:
                  counts[num] = 1
          
          keys = list(sorted(counts.keys()))
  
          max_count = 0
          for i in range(len(keys) - 1):
              k1 = keys[i]
              k2 = keys[i+1]
              if k2 - k1 == 1 and max_count < counts[k1] + counts[k2]:
                  max_count = counts[k1] + counts[k2]
  
          return max_count
  ```

* 枚举法（超出时间限制）

  ```python
  class Solution:
      def findLHS(self, nums: List[int]) -> int:
          
          max_count = 0
          for num in nums:
              c1 = 0
              c2 = 0
              for num0 in nums:
                  if num==num0:
                      c1 += 1
                  if (num + 1)==num0:
                      c2 += 1
                  
              if c1 > 0 and c2 > 0:
                  max_count = max(max_count, (c1+c2))
  
          return max_count
  ```

* 计数法2

  ```python
  class Solution:
      def findLHS(self, nums: List[int]) -> int:
          counts = {}
  
          for num in nums:
              if num in counts:
                  counts[num] = counts[num] + 1
              else:
                  counts[num] = 1
          
          max_count = 0
  
          for k in counts.keys():
              if (k + 1) in counts.keys():
                  max_count = max(max_count, counts[k] + counts[k+1])
  
          return max_count
  ```

### 23. 图片平滑器

* 暴力解法

  ```python
  class Solution:
      def imageSmoother(self, M: List[List[int]]) -> List[List[int]]:
          m, n = len(M), len(M[0])
          A = [[0 for _ in range(n)] for _ in range(m)]
  
          for i in range(m):
              for j in range(n):
                  sums = 0
                  count = 0
                  for dx in [-1, 0, 1]:
                      for dy in [-1, 0, 1]:
                          if i + dx >= 0 and i + dx < m and j + dy >=0 \
                          and j + dy < n:
                              sums = sums + M[i+dx][j+dy]
                              count = count + 1
  
                  A[i][j] = int(1.0*sums/count)
          
          return A
  ```

### 24. 种花问题

* 动态规划

  ```python
  class Solution:
      def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
          if len(flowerbed)==1:
              return 1-flowerbed[0] >= n
  
          opt = [0 for _ in range(len(flowerbed))]
          
          if flowerbed[0] + flowerbed[1]==0:
              opt[0] = 1
              opt[1] = 1
          else:
              opt[0] = 0
              opt[1] = 0
  
          for i in range(2, len(flowerbed)):
              if i < len(flowerbed) - 1:
                  if flowerbed[i] + flowerbed[i+1] + flowerbed[i-1] == 0:
                      opt[i] = max(1+opt[i-2], opt[i-1])
                  elif flowerbed[i]==1:
                      opt[i] = opt[i-2]
                  else:
                      opt[i] = opt[i-1]
              else:
                  if flowerbed[i] + flowerbed[i-1] == 0:
                      opt[i] = max(1+opt[i-2], opt[i-1])
                  elif flowerbed[i]==1:
                      opt[i] = opt[i-2]
                  else:
                      opt[i] = opt[i-1]
  
          return opt[len(flowerbed)-1] >= n
  
  ```

* 贪心算法

  ```python
  class Solution:
      def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
  
          if len(flowerbed)==1:
              return 1 - flowerbed[0] >= n
  
  
          count = 0
          for i in range(len(flowerbed)):
              if (i==0 and flowerbed[i]==0 and flowerbed[i+1]==0) or \
              (i==len(flowerbed)-1 and flowerbed[i]==0 and flowerbed[i-1]==0) or \
              (flowerbed[i-1]==0 and flowerbed[i]==0 and flowerbed[i+1]==0):
                  flowerbed[i] = 1
                  count = count + 1
  
          return count >= n
  ```

### 25. 两个列表的最小索引总和

* 暴力法

  ```python
  class Solution:
      def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
  
          min_sum = 2000
          rest = []
          for i, x in enumerate(list1):
              idx = list2.index(x) if x in list2 else -1
              if idx >= 0:
                  if min_sum==(i + idx):
                      rest.append(x)
                      
                  if min_sum > (i + idx):
                      if rest:
                          rest.pop()
                      rest.append(x)
                      min_sum = i + idx
            
          return rest
  ```

* 哈希表

  ```python
  class Solution:
      def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
  
          hashmap1 = dict(zip(list1, range(len(list1))))
          hashmap2 = dict(zip(list2, range(len(list2))))
  
          rest = []
          min_idx = 2000
          for k in hashmap1.keys():
              if k in hashmap2.keys():
                  sums = hashmap1[k] + hashmap2[k]
                  if min_idx > sums:
                      if rest:
                          rest.pop()
                      rest.append(k)
                      min_idx = sums
                  elif min_idx==sums:
                      rest.append(k)
  
          return rest
  ```

### 26. 错误的集合

* 排序

  ```python
  class Solution:
      def findErrorNums(self, nums: List[int]) -> List[int]:
  
          nums.sort()
          for i in range(len(nums)-1):
              if nums[i]==nums[i+1]:
                  dup = nums[i]
                  break
  
          sums = sum(list(range(1, len(nums)+1)))
  
          missing = sums - (sum(nums) - dup)
  
          return [dup, missing]
  ```

* hash

  ```python
  class Solution:
      def findErrorNums(self, nums: List[int]) -> List[int]:
  
          counts = {}
          for num in range(len(nums)):
              counts[num+1] = 0
  
          for num in nums:
              counts[num] += 1
  
          for k, v in counts.items():
              if v==2:
                  dup = k
              if v==0:
                  missing = k
  
          return [dup, missing]
  ```



### 27. 最长连续递增序列

* 暴力

  ```python
  class Solution:
      def findLengthOfLCIS(self, nums: List[int]) -> int:
  
          if len(nums)==0:
              return 0
  
          count = 0
          max_c = 0
  
          for i in range(len(nums)-1):
              if nums[i] >= nums[i+1]:
                  max_c = max(max_c, count)
                  count = 0
              else:
                  count += 1
          
  
          return max(count, max_c) + 1
  ```


### 28. 最短无序连续子数组

* 双指针

  ```python
  class Solution:
      def findUnsortedSubarray(self, nums: List[int]) -> int:
          if len(nums)==1:
              return 0
  
          left = 0
          while left < len(nums)-1 and nums[left] <= nums[left+1]:
              left = left + 1
  
          right = len(nums) - 1
          while right > 0 and nums[right] >= nums[right-1]:
              right = right - 1
  
          if left > right:
              return 0
  
          min_v = min(nums[left:right+1])
          max_v = max(nums[left:right+1])
  
          while left >= 0 and nums[left] > min_v:
              left = left - 1
          while right < len(nums) and nums[right] < max_v:
              right = right + 1
  
          left = 0 if left < 0 else left + 1
          right = right-1 if right < len(nums) else len(nums)-1
  
       return right - left + 1
  ```

### 29. 验证回文字符串II

```python
class Solution:

    def isPalindrome(self, s):
        if len(s)==1:
            return True
        
        i = 0
        j = len(s)-1
        while i < j and s[i]==s[j]:
            i += 1
            j -= 1
        
        return i >= j

    def validPalindrome(self, s: str) -> bool:
        
        i = 0
        j = len(s) - 1
        while i < j and s[i]==s[j]:
            i += 1
            j -= 1
        
        if i >= j:
            return True
        
        return self.isPalindrome(s[i: j]) or self.isPalindrome(s[i+1:j+1])

        
```

### 30. 非递减数列

* 暴力法

  ```python
  class Solution:
  
      def isPossibility(self, nums):
          
          for i in range(len(nums)-1):
              if nums[i] > nums[i+1]:
                  return False
          
          return True
  
      def checkPossibility(self, nums: List[int]) -> bool:
          
          if len(nums)==1:
              return True
  
          for i in range(len(nums)-1):
              if nums[i] > nums[i+1]:
                  if i==0:
                      nums[0] = nums[1]
                  elif i==len(nums)-2:
                      nums[-1] = nums[-2]
                  elif nums[i] > nums[i+2]:
                      nums[i] = nums[i+1]
                  else:
                      nums[i+1] = nums[i]
  
                  return self.isPossibility(nums)
  
          return True
                      
  ```

  

### 31. 重复叠加字符串匹配

* 暴力

  ```python
  class Solution:
      def repeatedStringMatch(self, A: str, B: str) -> int:
  
          max_len = 2*len(A) + len(B)
          C = A
          k = 1
          while len(A) < max_len and B not in A:
              A = A + C
              k += 1
          
          if B in A:
              return k
          else:
              return -1
  ```

### 32. 棒球比赛

```python
class Solution:
    def calPoints(self, ops: List[str]) -> int:

        stack = []
        sums = 0
        for i in range(len(ops)):
            if i==0:
                stack.append(int(ops[i]))
                res = int(ops[i])
            elif ops[i]=='C':
                res = stack.pop()
                res = -res
            elif ops[i]=='D':
                res = stack[-1]*2
                stack.append(res)
            elif ops[i]=='+':
                res = stack[-1] + stack[-2]
                stack.append(res)
            else:
                res = int(ops[i])
                stack.append(res)

            sums += res

        return sums
```

### 33. 交替位二进制数

* 移位法

  ```python
  class Solution:
      def hasAlternatingBits(self, n: int) -> bool:
  
          prev = n&1
          n = n >> 1
          while n:
              if n%2==prev:
                  return False
              else:
                  prev = n&1
                  n = n >> 1
          return True
  ```

* 转换成字符串

  ```python
  class Solution:
      def hasAlternatingBits(self, n: int) -> bool:
  
          nums = []
          while n:
              nums.append(n&1)
              n = n >> 1
          
          for i in range(len(nums)-1):
              if nums[i]==nums[i+1]:
                  return False
  
          return True
  ```

### 34. 计数二进制字串

* 连续数字统计

  ```python
  class Solution:
      def countBinarySubstrings(self, s: str) -> int:
  
          i = 0
  
          counts = []
          while i < len(s):
              j = i
              while j < len(s) and s[i]==s[j]:
                  j = j + 1
              
              counts.append(j-i)
              i = j
  
          if len(counts)==1:
              return 0
          
          sums = 0
          for i in range(len(counts)-1):
              sums += min(counts[i], counts[i+1])
  
          return sums
  ```

* 连续数字统计优化

  ```python
  class Solution:
      def countBinarySubstrings(self, s: str) -> int:
  
          i = 0
          counts = 0
          while i < len(s):
              j = i
              while j < len(s) and s[i]==s[j]:
                  j = j + 1
              
              if i==0:
                  prev = j - i
              else:
                  counts += min(prev, j-i)
                  prev = j - i
  
              i = j
  ```

### 35. 数组的度

* 三个哈希表

  ```python
  class Solution:
      def findShortestSubArray(self, nums: List[int]) -> int:
  
          left = {}
          right = {}
          counts = {}
  
          for i, x in enumerate(nums):
              if x not in left:
                  left[x] = i
                  counts[x] = 0
              
              right[x] = i
              counts[x] += 1
          
          d = max(counts.values())
          min_len = len(nums)
  
          for k, v in counts.items():
              if d==v:
                  min_len = min(min_len, right[k]-left[k]+1)
  
          return min_len
  ```

### 36. 存在重复元素

* 哈希表

  ```python
  class Solution:
      def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
  
  
          left = {}
          dist = {}
  
  
          for i, num in enumerate(nums):
              if num not in left:
                  left[num] = i
                  dist[num] = len(nums)
              else:
                  dist[num] = min(dist[num], i - left[num])
                  left[num] = i
      
          if len(nums) <= 1:
              return False
          else:
              min_len = min(dist.values())
              if min_len==len(nums):
                  return False
              return min_len <= k
  ```

### 37. 自除数

```python
class Solution:
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:

        divs = []

        for x in range(left, right+1):
            a = x
            b = x%10
            while a:
                if b==0 or x%b!=0:
                    break
                a = a//10
                b = a%10
            
            if a==0:
                divs.append(x)
        
        return divs
```

### 38. 二叉搜索树中的搜索

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def searchBST(self, root: TreeNode, val: int) -> TreeNode:
          if root==None:
              return None
          
          if root.val==val:
              return root
  
          left = self.searchBST(root.left, val)
          right = self.searchBST(root.right, val)
  
          if left==None:
              return right
          else:
              return left
  ```

* 递归加速

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def searchBST(self, root: TreeNode, val: int) -> TreeNode:
          if root==None:
              return None
          
          if root.val==val:
              return root
          elif root.val > val:
              return self.searchBST(root.left, val)
          else:
              return self.searchBST(root.right, val)
  ```

### 39. 二叉搜索树结点最小距离

* 中序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import sys
  class Solution:
  
      
      def dfs(self, root):
          if root==None:
              return
          if root.left:
              self.dfs(root.left)
  
          if self.prev==None:
              self.prev = root.val
          else:
              self.ans = min(self.ans, root.val-self.prev)
              self.prev = root.val
  
          if root.right:
              self.dfs(root.right)
  
  
      def minDiffInBST(self, root: TreeNode) -> int:
          self.ans = sys.maxsize
          self.prev = None
          self.dfs(root)
          return self.ans
  ```

* 排序

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import sys
  class Solution:
  
      
      def dfs(self, root):
          if root==None:
              return
          if root.left:
              self.dfs(root.left)
  
          self.values.append(root.val)
  
          if root.right:
              self.dfs(root.right)
  
  
      def minDiffInBST(self, root: TreeNode) -> int:
          self.values = []
          self.dfs(root)
          min_v = sys.maxsize
          for i in range(len(self.values)-1):
              min_v = min(min_v, self.values[i+1]- self.values[i])
  
          return min_v
  ```

* 排序（局部变量，速度稍慢)

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import sys
  class Solution:
  
      
      def dfs(self, root, values):
          if root==None:
              return
          if root.left:
              self.dfs(root.left, values)
  
          values.append(root.val)
  
          if root.right:
              self.dfs(root.right, values)
  
  
      def minDiffInBST(self, root: TreeNode) -> int:
          values = []
          self.dfs(root, values)
          min_v = sys.maxsize
          for i in range(len(values)-1):
              min_v = min(min_v, values[i+1]- values[i])
  
          return min_v
          
  ```

### 40. 叶子相似的树

* DFS

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, leaf):
          if root==None:
              return
          if root.left:
              self.dfs(root.left, leaf)
  
          if root.left==None and root.right==None:
              leaf.append(root.val)
  
          if root.right:
              self.dfs(root.right, leaf)
  
          
  
      def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
  
          leaf1 = []
          leaf2 = []
          self.dfs(root1, leaf1)
          self.dfs(root2, leaf2)
  
          if len(leaf1)!=len(leaf2):
              return False
  
          for i in range(len(leaf1)):
              if leaf1[i]!=leaf2[i]:
                  return False
          
          return True
  
  ```

### 41. 二叉搜索树的范围和

* DFS

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return
  
          if root.left:
              self.dfs(root.left)
  
          if root.val >= self.L and root.val <= self.R:
              self.sums += root.val
  
          if root.right:
              self.dfs(root.right)
  
      def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
          self.L, self.R = L, R
          self.sums = 0
  
          self.dfs(root)
  
          return self.sums
  ```

### 42. 单值二叉树

* DFS

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return
          if root.left:
              self.dfs(root.left)
          
          self.vals.append(root.val)
  
          if root.right:
              self.dfs(root.right)
  
  
      def isUnivalTree(self, root: TreeNode) -> bool:
          self.vals = []
  
          self.dfs(root)
  
          for i in range(len(self.vals)-1):
              if self.vals[i]!=self.vals[i+1]:
                  return False
  
          return True
  ```

* DFS优化

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return
          if root.left:
              self.dfs(root.left)
          
          if self.prev!=None and self.prev!=root.val:
              self.single=False
              return
          
          self.prev=root.val
  
          if root.right:
              self.dfs(root.right)
  
  
      def isUnivalTree(self, root: TreeNode) -> bool:
          self.prev=None
          self.single = True
  
          self.dfs(root)
  
          return self.single
  ```

### 43. 从根到叶的二进制数字之和

* 先序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, val):
          if root==None:
              return
  
          newval = val << 1 | root.val
  
          if root.left==None and root.right==None:
              self.sums += newval
          
          if root.left:
              self.dfs(root.left, newval)
          if root.right:
              self.dfs(root.right, newval)
  
      def sumRootToLeaf(self, root: TreeNode) -> int:
  
          self.sums = 0
          self.dfs(root, 0)
          return self.sums
  ```

### 44. 6和9组成的最大数字

* 字符串

  ```python
  class Solution:
      def maximum69Number (self, num: int) -> int:
          num = list(str(num))
          for i in range(len(num)):
              if num[i]=='6':
                  num[i]='9'
                  return int(''.join(num))
  
          return int(''.join(num))
  ```

### 45. 数组序号转换

* 哈希表

  ```python
  class Solution:
      def arrayRankTransform(self, arr: List[int]) -> List[int]:
          
          hashmap = {}
          array = sorted(list(set(arr)))
  
          for i, x in enumerate(array, 1):
              hashmap[x] = i
  
          return [hashmap[x] for x in arr]
  ```

### 46. 找出给定方程的正整数解

* 暴力法

  ```python
  """
     This is the custom function interface.
     You should not implement it, or speculate about its implementation
     class CustomFunction:
         # Returns f(x, y) for any given positive integers x and y.
         # Note that f(x, y) is increasing with respect to both x and y.
         # i.e. f(x, y) < f(x + 1, y), f(x, y) < f(x, y + 1)
         def f(self, x, y):
    
  """
  class Solution:
      def findSolution(self, customfunction: 'CustomFunction', z: int) -> List[List[int]]:
          x, y = 1, 1
  
          custom = customfunction
  
          ans = []
          while custom.f(x, y) < z:
  
              while custom.f(x, y) < z:
                  y = y + 1
  
              if custom.f(x, y)==z:
                  ans.append([x, y])
  
              y = 1
              x = x + 1
  
          if custom.f(x, y)==z:
              ans.append([x, y])
  
          return ans
  ```

### 47. 整数各位积和之差

* 暴力

  ```python
  class Solution:
      def subtractProductAndSum(self, n: int) -> int:
  
          sums = 0
          prod = 1
          while n:
              b = n%10
              n = n//10
              prod *= b
              sums += b
  ```

### 48. 统计位数为偶数的个数

* 字符串

  ```python
  class Solution:
      def findNumbers(self, nums: List[int]) -> int:
  
          ans = [len(str(num)) for num in nums]
  
          count = 0
          for x in ans:
              if x%2==0:
                  count += 1
  
          return count
  ```

* 减少空间占用

  ```python
  class Solution:
      def findNumbers(self, nums: List[int]) -> int:
  
          count = 0
          for num in nums:
              if len(str(num))%2==0:
                  count += 1
  
          return count
  ```

* 数值解法

  ```python
  class Solution:
      def findNumbers(self, nums: List[int]) -> int:
  
          count = 0
          for num in nums:
              c = 0
              while num > 0:
                  num = num//10
                  c += 1
              if c%2==0:
                  count += 1
  
          return count
  ```

### 49. 猜数字

```python
class Solution:
    def game(self, guess: List[int], answer: List[int]) -> int:

        count = 0
        for i in range(len(guess)):
            if guess[i]==answer[i]:
                count += 1

        return count
```

### 50. 字符串的最大公因子

* 笨方法

  ```python
  class Solution:
  
      def gcdOfStrings(self, str1: str, str2: str) -> str:
          
          a = len(str1)
          b = len(str2)
  
          while b:
              tmp = a%b
              a = b
              b = tmp
          
          while a:
              
              if str1[:a]*(len(str1)//a)==str1 and str1[:a]*(len(str2)//a)==str2:
                  return str1[:a]
  
              if a==1:
                  return ""
  
              i = 2
              while a%i!=0:
                  i += 1
              a = a//a
  ```

* 简单解法

  ```python
  class Solution:
  
  
      def gcd(self, a, b):
          return a if b==0 else self.gcd(b, a%b)
  
      def gcdOfStrings(self, str1: str, str2: str) -> str:
          
          if str1 + str2 != str2 + str1:
              return ""
  
          return str1[:self.gcd(len(str1), len(str2))]
  ```

### 51. Bigram分词

* 暴力

  ```python
  class Solution:
      def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
  
          i = 0
          ans = []
          while i < len(text):
              while i < len(text) and text[i]!=first[0]:
                  i += 1
  
              temp0 = i
  
              j = 0
              while i < len(text) and j < len(first) and text[i]==first[j]:
                  i += 1
                  j += 1
  
              if j==len(first) and i < len(text):
                  k = 0
                  i += 1
                  while i < len(text) and k < len(second) and text[i]==second[k]:
                      i += 1
                      k += 1
  
                  if k==len(second) and i < len(text):
                      i += 1
                      temp = i
                      while i < len(text) and text[i]!=' ':
                          i += 1
  
                      ans.append(text[temp: i])
              
              i = temp0
              while i < len(text) and text[i]!=' ':
                  i += 1
              i = i + 1
  
          return ans
  ```

* 匹配

  ```python
  class Solution:
      def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
  
          pattern = first + ' ' + second + ' '
  
          i = 0
          ans = []
          while i < len(text):
              while i < len(text) and pattern[0]!=text[i]:
                  i += 1
  
              temp0 = i
  
              j = 0
              while i < len(text) and j < len(pattern) and text[i]==pattern[j]:
                  i += 1
                  j += 1
              
              if i < len(text) and j==len(pattern):
                  temp = i
                  while i < len(text) and text[i]!=' ':
                      i += 1
  
                  ans.append(text[temp:i])
  
              i = temp0
              while i < len(text) and text[i]!=' ':
                  i += 1
              
  
          return ans
  ```

* 字符串分割

  ```python
  class Solution:
      def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
  
          sptext = text.split(' ')
  
          ans = []
          for i in range(len(sptext)-2):
              if sptext[i]==first and sptext[i+1]==second:
                  ans.append(sptext[i+2])
  
          return ans
  ```

### 52. 复写零

* 暴力

  ```python
  class Solution:
      def duplicateZeros(self, arr: List[int]) -> None:
          """
          Do not return anything, modify arr in-place instead.
          """
  
          i = 0
          while i < len(arr):
              
              while i < len(arr) and arr[i]!=0:
                  i += 1
  
              j = len(arr)-1
              while j > i:
                  arr[j] = arr[j-1]
                  j -= 1
              
              i += 2
  ```

* 统计0的个数

  ```python
  class Solution:
      def duplicateZeros(self, arr: List[int]) -> None:
          """
          Do not return anything, modify arr in-place instead.
          """
  
          idx = [0 for _ in range(len(arr))]
  
          c = 0
          for i in range(len(arr)):
              idx[i] = c
              if arr[i]==0:
                  c += 1
  
          print(idx)
          
          i = len(arr)-1
          while i >= 0:
              if i + idx[i] >= len(arr):
                  i -= 1
                  continue
  
              c = idx[i]
              arr[i+c] = arr[i]
              if arr[i]==0 and (i+c+1)<len(arr):
                  arr[i+c+1]=arr[i]
              i -= 1
          
  ```

* 临时数组

  ```python
  class Solution:
      def duplicateZeros(self, arr: List[int]) -> None:
          """
          Do not return anything, modify arr in-place instead.
          """
  
          newarr = [0 for _ in range(len(arr))]
          c = 0
  
          i = 0
          while i+c < len(arr):
              newarr[i+c] = arr[i]
              if arr[i]==0 and i+c+1 < len(arr):
                  newarr[i+c+1] = arr[i]
                  c += 1
              i += 1
  
          for i in range(len(arr)):
              arr[i] = newarr[i]
  ```

### 53. 分糖果II

* 暴力解法

  ```python
  class Solution:
      def distributeCandies(self, candies: int, num_people: int) -> List[int]:
  
          ans = [0 for _ in range(num_people)]
  
          c = 1
          i = 0
          while candies >= c:
              ans[i] += c
              candies -= c
              i = (i + 1)%num_people
              c = c + 1
          
          ans[i] += candies
  
          return ans
  ```

### 54. 数组的相对排序

* 哈希表

  ```python
  class Solution:
      def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
  
          counts = {}
  
          for x in arr1:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
          
          array = arr2 + sorted(list(set(arr1)-set(arr2)))
  
          ans = []
          for x in array:
              for _ in range(counts[x]):
                  ans.append(x)
          
          return ans
  ```

### 55. 等价多米诺骨牌对的数量

* 暴力解法（超时）

  ```python
  class Solution:
      def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
  
          count = 0
          for i in range(len(dominoes)):
              for j in range(len(dominoes)):
                  if i < j:
                      a, b = dominoes[i]
                      c, d = dominoes[j]
                      if a==c and b==d or a==d and b==c:
                          count += 1
  
          return count
  ```

* 删除法

  ```python
  class Solution:
      def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
  
          i = 0
          ans = 0
          while i < len(dominoes):
              a, b = dominoes[i]
              count = 1
              j = i + 1
              while j < len(dominoes):
                  c, d = dominoes[j]
                  if a==c and b==d or a==d and b==c:
                      count += 1
                      dominoes.pop(j)
                  else:
                      j += 1
              i += 1
              ans += (count-1)*count//2
  ```

* 哈希表

  ```python
  class Solution:
      def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
  
          counts = {}
          for i in range(len(dominoes)):
              a, b = dominoes[i]
              x = min(a, b)*10 + max(a, b)
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          ans = 0
          for count in counts.values():
              ans += count*(count-1)//2
  
          return ans
  ```

### 56. 第 N 个泰波那契数

* O(n)

  ```python
  class Solution:
      def tribonacci(self, n: int) -> int:
          
          if n <= 1:
              return n
          if n==2:
              return 1
  
          a, b, c = 0, 1, 1
          for _ in range(3, n+1):
              sums = a + b + c
              a, b, c = b, c, sums
          
          return sums
  ```

### 57. 一年中的第几天

```python
class Solution:
    def dayOfYear(self, date: str) -> int:
        monthes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        year, month, day = list(map(int, date.split('-')))

        ping = True
        if year%4==0 and year%100!=0 or year%400==0:
            ping = False

        sums = sum(monthes[:month-1]) if month > 0 else 0
        sums += day
        if ping==False and month > 2:
            sums += 1

        return sums
```

### 58. 一周中的第几天

```python
class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        monthes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        weeks = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        sums = sum(monthes[:month-1]) if month > 1 else 0
        sums += day
        sums += 1 if month > 2 and (year%4==0 and year%100!=0 or year%400==0) else 0

        for y in range(1971, year):
            if y%4==0 and y%100!=0 or y%400==0:
                sums += 366
            else:
                sums += 365

        return weeks[(sums+4)%7]
```

### 59. 拼写单词

* 哈希表

  ```python
  class Solution:
      def countCharacters(self, words: List[str], chars: str) -> int:
  
          counts = {}
          for c in chars:
              if c in counts:
                  counts[c] += 1
              else:
                  counts[c] = 1
  
          ans = 0
          for word in words:
              wcount = {}
              for w in word:
                  if w in wcount:
                      wcount[w] += 1
                  else:
                      wcount[w] = 1
  
              valid = True
              for k, v in wcount.items():
                  if k not in counts or v > counts[k]:
                      valid = False
                      break
  
              ans += len(word) if valid else 0
  
          return ans
  ```

### 60. 比较字符串最小字母出现频次

* 暴力

  ```python
  class Solution:
  
      def f(self, s):
          min_c = ord(s[0])
          count = 0
          for c in s:
              if ord(c)==min_c:
                  count += 1
              elif ord(c) < min_c:
                  count = 1
                  min_c = ord(c)
  
          return count
              
  
      def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
  
          fqueries = [self.f(q) for q in queries]
          fwords = [self.f(w) for w in words]
  
  
          counts = []
          for fq in fqueries:
              count = 0
              for fw in fwords:
                  if fw > fq:
                      count += 1
  
              counts.append(count)
  
          return counts
  
  ```

### 61. 质数排列

* 全排列

  ```python
  class Solution:
  
      def isPrime(self, x):
          i = 2
          while i*i <= x and x%i!=0:
              i += 1
          
          return i*i > x
  
      def numPrimeArrangements(self, n: int) -> int:
  
          if n <= 2:
              return 1
  
          count = 0
          for i in range(2, n+1):
              if self.isPrime(i):
                  count += 1
  
          p1 = 1
          for c in range(1, count+1):
              p1 *= c
  
          p2 = 1
          for c in range(1, n-count+1):
              p2 *= c
  
          
  
          return (p1*p2)%(10**9+7)
  ```


### 62. “气球” 的最大数量

* 计数法

  ```python
  import sys
  class Solution:
      def maxNumberOfBalloons(self, text: str) -> int:
  
          counts = {}
          for x in text:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          bal = {}
          for x in 'balloon':
              if x in bal:
                  bal[x] += 1
              else:
                  bal[x] = 1
  
          n_b = sys.maxsize
          for k, v in bal.items():
              if k not in counts or counts[k] < v:
                  return 0
              n_b = min(n_b, counts[k]//v)
  
          return n_b
  ```

* 计数非哈希

  ```python
  
  class Solution:
      def maxNumberOfBalloons(self, text: str) -> int:
  
          counts = [text.count(x) if x in 'ban' else text.count(x)//2 for x in 'balon']
          return min(counts)
  ```

### 63. 最小绝对差

* 排序

  ```python
  import sys
  class Solution:
      def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
  
          
          i = 0
          arr = sorted(arr)
          min_dist = sys.maxsize
          ans = []
          for i in range(len(arr)-1):
              if arr[i+1] - arr[i] < min_dist:
                  ans.clear()
                  min_dist = arr[i+1] - arr[i]
                  ans.append([arr[i], arr[i+1]])
              elif arr[i+1] - arr[i] == min_dist:
                  ans.append([arr[i], arr[i+1]])
  
          return ans
  ```

* 排序2

  ```python
  import sys
  class Solution:
      def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
  
          arr = sorted(arr)
  
          i = 0
          min_dist = sys.maxsize
  
          for i in range(len(arr)-1):
              if arr[i+1]-arr[i] < min_dist:
                  min_dist = arr[i+1]-arr[i]
          
          ans = []
          for i in range(len(arr)-1):
              if arr[i+1]-arr[i]==min_dist:
                  ans.append([arr[i], arr[i+1]])
  
          return ans
  ```

### 64. 独一无二的出现次数

* 哈希计数

  ```python
  class Solution:
      def uniqueOccurrences(self, arr: List[int]) -> bool:
  
          counts = {}
          for x in arr:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          return len(set(counts.values()))==len(list(counts.values()))
  ```

* 计数函数（稍慢）

  ```python
  class Solution:
      def uniqueOccurrences(self, arr: List[int]) -> bool:
  
          counts = [arr.count(x) for x in set(arr)]
          return len(counts) == len(set(counts))
  ```

### 65. 玩筹码

* 偶数奇数

  ```python
  class Solution:
      def minCostToMoveChips(self, chips: List[int]) -> int:
  
          odd = 0
          even = 0
          for x in chips:
              if x%2==0:
                  even += 1
              else:
                  odd += 1
          
          return min(odd, even)
  ```

* 暴力法

  ```python
  import sys
  class Solution:
      def minCostToMoveChips(self, chips: List[int]) -> int:
  
          counts = {}
          for x in chips:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
          
          cost = sys.maxsize
          for k, v in counts.items():
  
              sums = 0
              for k0, v0 in counts.items():
                  sums += v0*(abs(k-k0)%2)
  
              cost = min(sums, cost)
  
          return cost
  ```

### 66. 分割平衡字符串

* 暴力

  ```python
  class Solution:
      def balancedStringSplit(self, s: str) -> int:
  
          i = 0
          ans = 0
          while i < len(s):
              j = i
  
              count = 0
              while j < len(s):
                  count += 1 if s[j]=='R' else -1
  
                  if count==0:
                      ans += 1
                      break
  
                  j += 1
              
              i = j + 1
  
          return ans
  ```

* 暴力法非break

  ```python
  class Solution:
      def balancedStringSplit(self, s: str) -> int:
  
          i = 0
          ans = 0
          while i < len(s):
              j = i
  
              count = 0
              while j < len(s) and (count!=0 or j==i):
                  count += 1 if s[j]=='R' else -1
                  j += 1
  
              ans += 1
              i = j
              
  
          return ans
  ```

* 计数为0的次数

  ```python
  class Solution:
      def balancedStringSplit(self, s: str) -> int:
  
          i = 0
          count = 0
          ans = 0
          for x in s:
              if x=='L':
                  count += 1
              else:
                  count -= 1
  
              if count==0:
                  ans += 1
  
          return ans
  ```

### 67. 缀点成线

* 斜率相同

  ```python
  class Solution:
      def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
  
          dx0= coordinates[1][0] - coordinates[0][0]
          dy0= coordinates[1][1] - coordinates[0][1]
  
          for i in range(2, len(coordinates)):
              dx1 = coordinates[i][0] - coordinates[0][0]
              dy1 = coordinates[i][1] - coordinates[0][1]
  
              if dx0*dy1!=dy0*dx1:
                  return False
              
          return True
  ```

### 68. 访问所有点的最小时间

* 水平和垂直方向最大值

  ```python
  class Solution:
      def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
  
          if len(points)==1:
              return 0
  
          ans = 0
          for i in range(1, len(points)):
              dx = points[i][0] - points[i-1][0]
              dy = points[i][1] - points[i-1][1]
              ans += max(abs(dx), abs(dy))
  
          return ans
  ```

### 69. 找出井字棋的获胜者

* 暴力

  ```python
  class Solution:
      def tictactoe(self, moves: List[List[int]]) -> str:
  
          status = [[0 for _ in range(3)] for _ in range(3)]
  
          for index, (i, j) in enumerate(moves):
              if index%2==0:
                  status[i][j] = 1
              else:
                  status[i][j] = -1
  
              if sum(status[i])==3:
                  return 'A'
              elif sum(status[i])==-3:
                  return 'B'
              elif sum([status[k][j] for k in range(3)])==3:
                  return 'A'
              elif sum([status[k][j] for k in range(3)])==-3:
                  return 'B'
              elif sum([status[k][k] for k in range(3)])==3:
                  return 'A'
              elif sum([status[k][k] for k in range(3)])==-3:
                  return 'B'
              elif sum([status[k][2-k] for k in range(3)])==3:
                  return 'A'
              elif sum([status[k][2-k] for k in range(3)])==-3:
                  return 'B'
              
              
          if len(moves)==9:
              return 'Draw'
          else:
              return 'Pending'
  ```

### 70. 分式化简

* 暴力

  ```python
  class Solution:
  
      def gcd(self, a, b):
          return a if b==0 else self.gcd(b, a%b)
  
      def fraction(self, cont: List[int]) -> List[int]:
  
          a, b = cont[-1], 1
          
          i = len(cont)-2
          while i >= 0:
              c = cont[i]
              temp = a
              a = a*c + b
              b = temp
              i -= 1
  		# 实际不需要约分
          gc = self.gcd(a, b)
          return [a//gc, b//gc]
  ```

### 71. 删除回文子序列

* 先删除a，再删除b

  ```python
  class Solution:
      def removePalindromeSub(self, s: str) -> int:
  
          if s=="":
              return 0
  
          i = 0
          j = len(s)-1
          while i < j and s[i]==s[j]:
              i += 1
              j -= 1
  
          if i >= j:
              return 1
          else:
              return 2
  ```

* 极简

  ```python
  class Solution:
      def removePalindromeSub(self, s: str) -> int:
  
          if s=='':
              return 0
          elif s==s[::-1]:
              return 1
          else:
              return 2
  ```

### 72. 三角形最大周长

* 排序

  ```python
  class Solution:
      def largestPerimeter(self, A: List[int]) -> int:
  
          A = sorted(A)
  
          i = len(A)-1
          while i >= 2 and A[i] >= (A[i-1] + A[i-2]):
              i -= 1
          
          if i >= 2:
              return A[i] + A[i-1] + A[i-2]
          else:
              return 0
  ```


### 73. 将每个元素替换为右侧最大元素

* 右端开始

  ```python
  class Solution:
      def replaceElements(self, arr: List[int]) -> List[int]:
          if len(arr)==1:
              return [-1]
  
          max_v = -1
          i = len(arr)-1
          while i >= 0:
              temp = arr[i]
              arr[i] = max_v
              max_v = max(max_v, temp)
              i -= 1
  
          return arr
  ```

### 74. 有序数组中出现次数超过25%的元素

* 双指针

  ```python
  class Solution:
      def findSpecialInteger(self, arr: List[int]) -> int:
          
          if len(arr)==1:
              return arr[0]
  
          i = 0
          j = i
          while j < len(arr):
              if arr[i]!=arr[j]:
                  if (j-i)*4 > len(arr):
                      return arr[i]
                  else:
                      i = j
              else:
                  j += 1
              
          if (j-i)*4 > len(arr):
              return arr[i]
  ```

* 单指针

  ```python
  class Solution:
      def findSpecialInteger(self, arr: List[int]) -> int:
          
          if len(arr)==1:
              return arr[0]
  
          span = len(arr)//4 + 1 if len(arr)%4!=0 else len(arr)//4
          j = 0
          while (j+span) < len(arr):
              if arr[j]==arr[j+span]:
                  return arr[j]
              else:
                  j += 1
  ```

### 75. 方阵中战斗力最弱的 K 行

* 索引绑定排序

  ```python
  class Solution:
      def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
  
          power = [sum(row) for row in mat]
          index = list(range(len(power)))
          pow_idx = list(zip(power, index))
          pow_idx = sorted(pow_idx, key=lambda x: x[0])
  
          idx = []
          for i in range(k):
              idx.append(pow_idx[i][-1])
  
          return idx
  ```

* 双权重

  ```python
  class Solution:
      def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
  
          weight = [sum(row) for row in mat]
  
          weight = [x*len(mat)+i for i, x in enumerate(weight)]
          weight.sort()
  
          idx = []
          for i in range(k):
              idx.append(weight[i]%(len(mat)))
  
          return idx
  ```

### 76. 将整数转换为两个无零整数的和

* 枚举

  ```python
  class Solution:
      def getNoZeroIntegers(self, n: int) -> List[int]:
  
          for a in range(1, n):
              b = n - a
              if '0' not in str(a) and '0' not in str(b):
                  return [a, b]
  ```

* 字符串

  ```python
  class Solution:
      def getNoZeroIntegers(self, n: int) -> List[int]:
  
          ans = ''
          temp = n
          while n:
              if n%10==1:
                  ans = '2' + ans
                  n = (n-2)//10
              else:
                  ans = '1' + ans
                  n = (n-1)//10
  
              if n < 10:
                  break
  
              
          return [int(ans), temp-int(ans)]
  ```

### 77. 解压缩编码列表

* 暴力

  ```python
  class Solution:
      def decompressRLElist(self, nums: List[int]) -> List[int]:
  
          ans = []
          for i in range(0, len(nums), 2):
              for _ in range(nums[i]):
                  ans.append(nums[i+1])
          
          return ans
  ```

### 78. 解码字母到整数映射

* 哈希表

  ```python
  
  class Solution:
      def freqAlphabets(self, s: str) -> str:
  
          lc = 'abcdefghijklmnopqrstuvwxyz'
          nums = [str(i) if i < 10 else str(i)+'#' for i in range(1, 27)]
          charmap = dict(zip(nums, lc))
  
          i = 0
          ans = []
          while i < len(s):
              if (i+2) < len(s) and s[i:i+3] in charmap:
                  ans.append(charmap[s[i:i+3]])
                  i += 3
              else:
                  ans.append(charmap[s[i]])
                  i += 1
              
  
          return ''.join(ans)
  ```

### 79. 二进制链表转整数

* 遍历

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def getDecimalValue(self, head: ListNode) -> int:
  
          val = 0
          while head:
              val = val*2 + head.val
              head = head.next
          
          return val
  ```

### 80. 奇数值单元格的数目

* 简单方法

  ```python
  class Solution:
      def oddCells(self, n: int, m: int, indices: List[List[int]]) -> int:
  
          row = []
          col = []
  
          for i, j in indices:
              if i in row:
                  row.remove(i)
              else:
                  row.append(i)
  
              if j in col:
                  col.remove(j)
              else:
                  col.append(j)
  
          return m*len(row)+n*len(col)-len(col)*len(row)*2
  ```

* 暴力

  ```python
  class Solution:
      def oddCells(self, n: int, m: int, indices: List[List[int]]) -> int:
  
          mat = [[0 for _ in range(m)] for _ in range(n)]
  
          for ri, ci in indices:
              for i in range(m):
                  mat[ri][i] = 1 - mat[ri][i]
  
              for j in range(n):
                  mat[j][ci] = 1 - mat[j][ci]
  
          return sum([sum(x) for x in mat])
  ```

### 81. 二维网格迁移

* 转置

  ```python
  class Solution:
      def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
  
          grid = [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
  
          for _ in range(k):
              temp = grid.pop()
              temp.insert(0, temp.pop())
              grid.insert(0, temp)
  
          grid = [[grid[i][j] for i in range(len(grid))] for j in range(len(grid[0]))]
  
          return grid
  ```

### 82. 删除字符串中所有相邻重复项

* 列表

  ```python
  class Solution:
      def removeDuplicates(self, S: str) -> str:
  
          i = 0
          S = list(S)
          while i < len(S)-1 and i >= 0:
              if S[i]==S[i+1]:
                  S.pop(i)
                  S.pop(i)
                  i = 0 if i==0 else i-1
              else:
                  i += 1
  
          return ''.join(S)
  ```

* 栈

  ```python
  class Solution:
      def removeDuplicates(self, S: str) -> str:
  
          ans = []
          
          for s in S:
              if len(ans)==0 or ans[-1]!=s:
                  ans.append(s)
              else:
                  ans.pop()
  
          return ''.join(ans)
  ```

### 83. 最后一块石头的重量

* 暴力排序

  ```python
  class Solution:
      def lastStoneWeight(self, stones: List[int]) -> int:
  
          while len(stones) > 1:
              stones.sort()
              if stones[-1]==stones[-2]:
                  stones.pop()
                  stones.pop()
              else:
                  stones[-2] = stones[-1] - stones[-2]
                  stones.pop()
              
          if len(stones)==1:
              return stones[-1]
          else:
              return 0
  ```


### 84. 两句话中不常见单词

* 集合

  ```python
  class Solution:
      def uncommonFromSentences(self, A: str, B: str) -> List[str]:
          tempA = A.split(' ')
          tempB = B.split(' ')
          A = set(tempA)
          B = set(tempB)
  
          ans = []
          for x in ((A|B) - (A&B)):
              if tempA.count(x)<2 and tempB.count(x)<2:
                  ans.append(x)
  
          return ans
  ```

* 哈希表

  ```python
  class Solution:
      def uncommonFromSentences(self, A: str, B: str) -> List[str]:
          
          A = A.split(' ')
          B = B.split(' ')
  
          countA = {}
          for x in A:
              if x in countA:
                  countA[x] += 1
              else:
                  countA[x] = 1
  
          countB = {}
          for x in B:
              if x in countB:
                  countB[x] += 1
              else:
                  countB[x] = 1
          ans = []
          for x in set(A)|set(B):
              if x in countA and countA[x]==1 and x not in countB:
                  ans.append(x)
              elif x in countB and countB[x]==1 and x not in countA:
                  ans.append(x)
          
  
          return ans
  ```

* 一个hash表

  ```python
  class Solution:
      def uncommonFromSentences(self, A: str, B: str) -> List[str]:
          
          s = A + ' ' + B
          s = s.split(' ')
  
          count = {}
          for x in s:
              if x in count:
                  count[x] += 1
              else:
                  count[x] = 1
          
          ans = []
          for k, v in count.items():
              if v==1:
                  ans.append(k)
  
          return ans
  ```

### 85. 糖果的公平交换

* 糖果总和差的一半

  ```python
  class Solution:
      def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
  
          dist = (sum(A) - sum(B))//2
  
          sa = set(A)
          sb = set(B)
          for a in sa:
              if int(a-dist) in sb:
                  return [a, a-dist]
  ```

### 86. 三维形体的表面积

* 暴力

  ```python
  class Solution:
      def surfaceArea(self, grid: List[List[int]]) -> int:
  
  
          n_items = sum([sum(g) for g in grid])
  
          sums = 0
          for i in range(len(grid)):
              for j in range(len(grid[0])-1):
                  minh = min(grid[i][j], grid[i][j+1])
                  sums += minh
          
          for j in range(len(grid[0])):
              for i in range(len(grid)-1):
                  minh = min(grid[i][j], grid[i+1][j])
                  sums += minh
  
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j] > 0:
                      sums += grid[i][j]-1
  
          return 6*n_items - 2*sums
  ```

### 87. 特殊等价字符串组

* 排序

  ```python
  class Solution:
  
      def numSpecialEquivGroups(self, A: List[str]) -> int:
  
          ans = set()
          for a in set(A):
              even = [a[i] for i in range(0, len(a), 2)]
              even.sort()
              odd = [a[i] for i in range(1, len(a), 2)]
              odd.sort()
  
              sums = []
              for i in range(len(a)):
                  if i%2==0:
                      sums.append(even[i//2])
                  else:
                      sums.append(odd[i//2])
  
              ans.add(''.join(sums))
  
          return len(ans)
  ```

### 88. 单调数列

* 一次遍历

  ```python
  class Solution:
      def isMonotonic(self, A: List[int]) -> bool:
  
          i = 0
          flag = 0
          while i < len(A)-1:
              if A[i]==A[i+1]:
                  i += 1
              elif A[i] < A[i+1]:
                  i += 1
                  if flag==-1:
                      return False
                  else:
                      flag = 1
              elif A[i] > A[i+1]:
                  i += 1
                  if flag==1:
                      return False
                  else:
                      flag = -1
  
          return True
  ```

  

* 两次遍历

  ```python
  class Solution:
      def isMonotonic(self, A: List[int]) -> bool:
  
          i = 0
          while i < len(A)-1 and A[i] >= A[i+1]:
              i += 1
  
          if i >= len(A)-1:
              return True
  
          i = 0
          while i < len(A)-1 and A[i] <= A[i+1]:
              i += 1
  
          if i >= len(A)-1:
              return True
          else:
             return False
  ```

### 89. 递增顺序查找树

* 先遍历，后建树

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __inTreeNodeit__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return
          
          if root.left:
              self.dfs(root.left)
  
          self.vals.append(root.val)
  
          if root.right:
              self.dfs(root.right)
  
      def increasingBST(self, root: TreeNode) -> TreeNode:
  
          self.vals = []
          self.dfs(root)
  
          tree = TreeNode(0)
          p = tree
          for i in range(len(self.vals)-1):
              p.val = self.vals[i]
              p.right = TreeNode(0)
              p = p.right
          
          p.val = self.vals[-1]
  
          return tree
  ```

### 90. 按奇偶排序数组

* 双指针

  ```python
  class Solution:
      def sortArrayByParity(self, A: List[int]) -> List[int]:
  
          i = 0
          j = len(A)-1
          while i < j:
              
              while i < j and A[i]%2==0:
                  i += 1
              
              while i < j and A[j]%2!=0:
                  j -= 1
              
              if i < j:
                  temp = A[i]
                  A[i] = A[j]
                  A[j] = temp
  
          return A
  ```

* 按照模2的结果排序

  ```python
  class Solution:
      def sortArrayByParity(self, A: List[int]) -> List[int]:
  
          return sorted(A, key=lambda x: x%2)
                  
  ```

### 91. 最小差值I

* 最小值区间和最大值区间

  ```python
  class Solution:
      def smallestRangeI(self, A: List[int], K: int) -> int:
  
          min_v, max_v = min(A), max(A)
          min_left, min_right = min_v - abs(K), min_v + abs(K)
          max_left, max_right = max_v - abs(K), max_v + abs(K)
  
          if min_right >= max_left:
              return 0
          else:
              return max_left - min_right
  ```

### 92. 卡牌分组

* 最大公约数

  ```python
  class Solution:
  
      def gcd(self, a, b):
  
          return a if b==0 else self.gcd(b, a%b)
  
      def hasGroupsSizeX(self, deck: List[int]) -> bool:
          counts = {}
          for d in deck:
              if d in counts:
                  counts[d] += 1
              else:
                  counts[d] = 1
          
          g = counts[deck[0]]
          for v in counts.values():
              g = self.gcd(g, v)
              if g < 2:
                  return False
  
          return True
  ```

### 93. 仅仅反转字母

* 双指针

  ```python
  class Solution:
      def reverseOnlyLetters(self, S: str) -> str:
  
          S = list(S)
          i = 0
          j = len(S)-1
          while i < j:
              
              while i < j and not S[i].isalpha():
                  i += 1
              while i < j and not S[j].isalpha():
                  j -= 1
              
              if i < j:
                  temp = S[i]
                  S[i] = S[j]
                  S[j] = temp
                  i += 1
                  j -= 1
  
          return ''.join(S)
  ```

* 字母栈顶

  ```python
  class Solution:
      def reverseOnlyLetters(self, S: str) -> str:
  
          stk = []
          for s in S:
              if s.isalpha():
                  stk.append(s)
  
          ans = []
          for s in S:
              if s.isalpha():
                  ans.append(stk.pop())
              else:
                  ans.append(s)
  
          return ''.join(ans)
  ```

### 94. 长按键入

* 双指针

  ```python
  class Solution:
      def isLongPressedName(self, name: str, typed: str) -> bool:
  
          i = 0
          j = 0
          count = 0
          while i < len(name) and j < len(typed):
              
              v = name[i]
              temp = i
              while i < len(name) and name[i]==v:
                  i += 1
              count1 = i-temp
  
              temp = j
              while j < len(typed) and typed[j]==v:
                  j += 1
              count2 = j-temp
  
              if count2 < count1:
                  return False
              elif count2 > count1:
                  count += 1
  
          if i >= len(name) and j >= len(typed) and count >= 0:
              return True
          else:
              return False
  ```

* 集合

  ```python
  class Solution:
      def numUniqueEmails(self, emails: List[str]) -> int:
  
          left = [email.split('@', 1)[0] for email in emails]
          right = [email.split('@', 1)[-1] for email in emails]
  
          left = [em.split('+')[0] if '+' in em else em for em in left]
          left = [em.replace('.', '') for em in left]
  
          emails = [left[i] + '@' + right[i] for i in range(len(left))]
          return len(set(emails))
  ```

### 95. 重新排列日志文件

* 选择排序

  ```python
  class Solution:
  
      def comparelog(self, a, b):
          aleft, aright = a.split(' ', 1)
          bleft, bright = b.split(' ', 1)
  
          if aright == bright:
              return aleft > bleft
          else:
              return aright > bright
  
      def reorderLogFiles(self, logs: List[str]) -> List[str]:
  
          alpha = [log for log in logs if log[-1].isalpha()]
          nums = [log for log in logs if not log[-1].isalpha()]
  
          for i in range(len(alpha)):
              index = i
              for j in range(i, len(alpha)):
                  if self.comparelog(alpha[index], alpha[j]):
                      index = j
              
              temp = alpha[i]
              alpha[i] = alpha[index]
              alpha[index] = temp
          
          return alpha + nums
  ```

* 自定义排序

  ```python
  import functools
  class Solution:
  
      def compare(self, a, b):
          aleft, aright = a.split(' ', 1)
          bleft, bright = b.split(' ', 1)
  
          if aright == bright and aleft > bleft or aright > bright:
              return 1
          elif aright==bright and aleft < bleft or aright < bright:
              return -1
          else:
              return 0
              
      def reorderLogFiles(self, logs: List[str]) -> List[str]:
  
          alpha = [log for log in logs if log[-1].isalpha()]
          nums = [log for log in logs if not log[-1].isalpha()]
  
          alpha = sorted(alpha, key = functools.cmp_to_key(self.compare))
          
          return alpha + nums
  ```

  

### 96. 有效的山脉数组

* 暴力

  ```python
  class Solution:
      def validMountainArray(self, A: List[int]) -> bool:
  
          if len(A) < 3:
              return False
  
          i = 0
          while i < len(A)-1 and A[i] < A[i+1]:
              i += 1
  
          if i >= len(A)-1 or i==0:
              return False
  
  
          while i < len(A)-1 and A[i] > A[i+1]:
              i += 1
  
          if i < len(A)-1:
              return False
          else:
              return True
  ```

### 97. 增减字符串匹配

* 官方

  ```python
  class Solution:
      def diStringMatch(self, S: str) -> List[int]:
  
          nums = list(range(len(S)+1))
  
          ans = []
          for s in S:
              if s=='I':
                  ans.append(nums[0])
                  nums.pop(0)
              else:
                  ans.append(nums[-1])
                  nums.pop()
  
          ans.append(nums.pop())
  
          return ans
  ```

* 双指针值

  ```python
  class Solution:
      def diStringMatch(self, S: str) -> List[int]:
  
          ans = []
          min_v = 0
          max_v = len(S)
          for s in S:
              if s=='I':
                  ans.append(min_v)
                  min_v += 1
              else:
                  ans.append(max_v)
                  max_v -= 1
              
          ans.append(min_v)
  
          return ans
  
  ```

### 98. 删列造序

* 计数

  ```python
  class Solution:
      def minDeletionSize(self, A: List[str]) -> int:
  
          count = 0
          for i in range(len(A[0])):
              for j in range(len(A)-1):
                  if A[j][i] > A[j+1][i]:
                      count += 1
                      break
          
          return count
  ```

### 99. 给定数字能组成的最大时间

* 全排列递归

  ```python
  class Solution:
  
      def perm(self, A, i):
          if i==len(A)-1:
              t = '{}{}:{}{}'.format(*A)
              if t[:2]<='23' and t[-2:]<='59':
                  self.ans.append(t)
              return
                  
  
          for j in range(i, len(A)):
              temp = A[j]
              A[j] = A[i]
              A[i] = temp
  
              self.perm(A, i+1)
  
              temp = A[j]
              A[j] = A[i]
              A[i] = temp
  
  
      def largestTimeFromDigits(self, A: List[int]) -> str:
  
          self.ans = []
          self.perm(A, 0)
          if len(self.ans)==0:
              return ""
  
          return max(self.ans)
  ```

### 100. 有序数组的平方

* 排序

  ```python
  class Solution:
      def sortedSquares(self, A: List[int]) -> List[int]:
  
          A = [x**2 for x in A]
          A.sort()
          return A
  ```

### 101. 重复N次的元素

* 集合

  ```python
  class Solution:
      def repeatedNTimes(self, A: List[int]) -> int:
  
          counts = set()
          for x in A:
              if x in counts:
                  return x
              else:
                  counts.add(x)
  ```

* 比较

  ```python
  class Solution:
      def repeatedNTimes(self, A: List[int]) -> int:
  
          for dist in range(1, len(A)):
              for i in range(len(A)):
                  if (i+dist) >= len(A):
                      break
                  if A[i]==A[i+dist]:
                      return A[i]
  ```

### 102. 强整数

* 暴力

  ```python
  class Solution:
      def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
  
          i = 0
          j = 0
          ans = set()
          while x**i + y**j <= bound:
              
              while x**i + y**j <= bound:
                  ans.add(x**i + y**j)
                  
                  if y==1:
                      break
                  j += 1
              
              if x==1:
                  break
              i += 1
              j = 0
  
          return list(ans)
  ```

### 103. 数组形式的整数加法

* 字符串法

  ```python
  class Solution:
      def addToArrayForm(self, A: List[int], K: int) -> List[int]:
  
          A = int(''.join(map(str, A)))
          return list(map(int, list(str(A+K))))
  ```

* 数值解法

  ```python
  class Solution:
      def addToArrayForm(self, A: List[int], K: int) -> List[int]:
  
          sums = 0
          for a in A:
              sums = 10*sums + a
          
          sums += K
  
          if sums==0:
              return [0]
  
          ans = []
          while sums:
              ans.append(sums%10)
              sums //= 10
          
          return ans[::-1]
  ```

### 104. 找到小镇的法官

* 计数法

  ```python
  class Solution:
      def findJudge(self, N: int, trust: List[List[int]]) -> int:
  
          if N==1 and len(trust)==0:
              return 1
  
          right = {}
          left = {}
          for x, y in trust:
              if y in right:
                  right[y] += 1
              else:
                  right[y] = 1
              
              if x in left:
                  left[x] += 1
              else:
                  left[x] = 1
  
          for k, v in right.items():
              if v==N-1 and k not in left:
                  return k
      
          return -1
  ```


### 105. 腐烂的橘子

* 栈

  ```python
  class Solution:
      def orangesRotting(self, grid: List[List[int]]) -> int:
  
          t = 0
          while True:
  
              rot = []
              count1 = 0
              for i in range(len(grid)):
                  for j in range(len(grid[0])):
                      if grid[i][j]==2:
                          rot.append((i, j))
                      elif grid[i][j]==1:
                          count1 += 1
              
              if count1==0:
                  return t
              
              count = 0
              while len(rot)>0:
                  i, j = rot.pop()
                  if (j-1)>=0 and grid[i][j-1]==1:
                      grid[i][j-1] = 2
                      count += 1
                  if (j+1)<len(grid[0]) and grid[i][j+1]==1:
                      grid[i][j+1] = 2
                      count += 1
                  if (i+1)<len(grid) and grid[i+1][j]==1:
                      grid[i+1][j] = 2
                      count += 1
                  if (i-1)>=0 and grid[i-1][j]==1:
                      grid[i-1][j] = 2
                      count += 1
  
              if count!=0:
                  t += 1
              elif count1!=0:
                  return -1
              
  
          return t
  ```

* 广度优先遍历

  ```python
  import queue
  class Solution:
      def orangesRotting(self, grid: List[List[int]]) -> int:
  
          q = queue.Queue()
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j]==2:
                      q.put((i, j))
  
          count = 0
          while not q.empty():
  
              lens = q.qsize()
              for _ in range(lens):
                  i, j = q.get()
                  if (j-1)>=0 and grid[i][j-1]==1:
                      grid[i][j-1] = 2
                      q.put((i, j-1))
                  if (i+1)<len(grid) and grid[i+1][j]==1:
                      grid[i+1][j] = 2
                      q.put((i+1, j))
                      count
                  if (j+1)<len(grid[0]) and grid[i][j+1]==1:
                      grid[i][j+1] = 2
                      q.put((i, j+1))
                  if (i-1)>=0 and grid[i-1][j]==1:
                      grid[i-1][j] = 2
                      q.put((i-1, j))
              
              if not q.empty():
                  count += 1
  
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j]==1:
                      return -1
  
          return count
  ```

### 106. 车的可捕获量

* 暴力

  ```python
  class Solution:
      def numRookCaptures(self, board: List[List[str]]) -> int:
          ans = 0
          for i in range(len(board)):
              for j in range(len(board[0])):
                  if board[i][j]=='R':
                      r, c = i, j
  
          j = c-1
          while j >= 0 and board[r][j]=='.':
              j -= 1
          if j >= 0 and board[r][j]=='p':
              ans += 1
          
          j = c+1
          while j < len(board[0]) and board[r][j]=='.':
              j += 1
          if j < len(board[0]) and board[r][j]=='p':
              ans += 1
          
          i = r-1
          while i >= 0 and board[i][c]=='.':
              i -= 1
          if i >= 0 and board[i][c]=='p':
              ans += 1
  
          i = r + 1
          while i < len(board) and board[i][c]=='.':
              i += 1
          if i < len(board) and board[i][c]=='p':
              ans += 1
  
          return ans
  ```

### 107. 查找常用字符串

* 计数法

  ```python
  class Solution:
      def commonChars(self, A: List[str]) -> List[str]:
  
          count = {}
          for x in A[0]:
              if x in count:
                  count[x] += 1
              else:
                  count[x] = 1
  
          for i in range(1, len(A)):
              temp = {}
              for x in A[i]:
                  if x in temp:
                      temp[x] += 1
                  else:
                      temp[x] = 1
  
              for k, v in count.items():
                  if k in temp:
                      count[k] = min(v, temp[k])
                  else:
                      count[k] = 0
          ans = []
          for k, v in count.items():
              for _ in range(v):
                  ans.append(k)
  
          return ans
  ```

### 108. K 次取反后最大化的数组和

* 分情况讨论

  ```python
  class Solution:
      def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
  
          pos = [v for v in A if v >= 0]
          neg = [v for v in A if v < 0]
          if len(neg) >= K:
              neg = sorted(neg)
              for i in range(K):
                  neg[i] = -neg[i]
  
              return sum(pos) + sum(neg)
  
          
          n_change = K - len(neg)
          A = [abs(x) for x in A]
          if n_change%2==0:
              return sum(A)
          else:
              return sum(A)-2*min(A)
  ```

* 最小值取反

  ```python
  class Solution:
      def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
  
          for _ in range(K):
              A.sort()
              A[0] = -A[0]
  
          return sum(A)
  ```

### 109. 十进制整数的反码

* 数值计算

  ```python
  class Solution:
      def bitwiseComplement(self, N: int) -> int:
  
          if N==0:
              return 1
  
          ans = []
          while N:
              ans.append(1-N%2)
              N //= 2
  
          ans = list(reversed(ans))
          
          sums = 0
          for x in ans:
              sums = sums*2 + x
  
          return int(sums)
  ```

### 110. 总时间可被60整除的歌曲

* 组合

  ```python
  class Solution:
      def numPairsDivisibleBy60(self, time: List[int]) -> int:
          
          counts = {}
          for t in time:
              t = t%60
              if t in counts:
                  counts[t] += 1
              else:
                  counts[t] = 1
      
          ans = 0
          if 0 in counts:
              ans += counts[0]*(counts[0]-1)//2
          if 30 in counts:
              ans += counts[30]*(counts[30]-1)//2
  
          for i in range(1, 30):
              if i in counts and (60-i) in counts:
                  ans += counts[i]*counts[60-i]
  
          return ans
  ```

### 111. 将数组分成相等的三个部分

* 扫描

  ```python
  class Solution:
      def canThreePartsEqualSum(self, A: List[int]) -> bool:
  
          sums = sum(A)
          if sums%3!=0:
              return False
  
          sums0 = 0
          count1 = 0
          count2 = 0
          for x in A:
              sums0 += x
              if sums0==(sums//3):
                  count1 += 1
  
              if count1 > 0 and sums0==(sums//3*2):
                  count2 += 1
  
          return count1 > 0 and count2 > 0
  ```

### 112. 删除最外层的括号

* 一次扫描

  ```python
  class Solution:
      def removeOuterParentheses(self, S: str) -> str:
  
          ans = ['']
          count = 0
          for s in S:
  
              if count==0 and s=='(':
                  count += 1
              elif count > 0 and s=='(':
                  count += 1
                  ans.append(s)
              elif count > 0 and s==')':
                  count -= 1
                  if count > 0:
                      ans.append(s)
  
          return ''.join(ans)
  ```

### 113. 除数博弈

* 动态规划

  ```python
  class Solution:
      def divisorGame(self, N: int) -> bool:
  
          if N==1:
              return False
  
          if N==2:
              return True
  
          opt = [0 for _ in range(N+1)]
          opt[1] = 0
          opt[2] = 1
          for i in range(3, N+1):
              
              for b in range(1, i):
                  if i%b==0 and opt[i]==0:
                      opt[i] = 1 - opt[i-b]
                      if opt[i]==1:
                          break
              
          return opt[N]==1
  ```

* 找规律

  ```python
  class Solution:
      def divisorGame(self, N: int) -> bool:
  
          return N%2==0
  ```

### 114. 可被5整除的二进制前缀

* 扫描

  ```python
  class Solution:
      def prefixesDivBy5(self, A: List[int]) -> List[bool]:
  
          sums = 0
          ans = []
          for i in range(len(A)):
              sums = sums*2 + A[i]
              if sums%5==0:
                  ans.append(True)
              else:
                  ans.append(False)
  
          return ans
  ```

### 115. 两地调度

* 动态规划递归版(超时)

  ```python
  class Solution:
  
      def dp(self, costs, i, j):
          if i==0 and j==0:
              return 0
  
          if i > 0 and j > 0:
              opt1 = costs[i+j-1][0] + self.dp(costs, i-1, j)
              opt2 = costs[i+j-1][1] + self.dp(costs, i, j-1)
              return min(opt1, opt2)
          
          if i > 0 and j==0:
              return costs[i+j-1][0] + self.dp(costs, i-1, j)
          if j > 0 and i==0:
              return costs[i+j-1][1] + self.dp(costs, i, j-1)
  
  
      def twoCitySchedCost(self, costs: List[List[int]]) -> int:
  
          return self.dp(costs, len(costs)//2, len(costs)//2)
  ```

* 动态规划

  ```python
  class Solution:
  
      def twoCitySchedCost(self, costs: List[List[int]]) -> int:
  
          r = len(costs)//2
          opt = [[0 for _ in range(r+1)] for _ in range(r+1)]
          
          for j in range(1, r+1):
              opt[0][j] = opt[0][j-1] + costs[j-1][1]
  
          for i in range(1, r+1):
              opt[i][0] = opt[i-1][0] + costs[i-1][0]
  
          for i in range(1, r+1):
              for j in range(1, r+1):
                  opt1 = costs[i+j-1][0] + opt[i-1][j]
                  opt2 = costs[i+j-1][1] + opt[i][j-1]
                  opt[i][j] = min(opt1, opt2)
                  
          return opt[r][r]
  ```

* 贪心算法

  ```python
  class Solution:
  
      def twoCitySchedCost(self, costs: List[List[int]]) -> int:
  
          sums = 0
          diff = []
          for cost in costs:
              sums += cost[1]
              diff.append(cost[0]-cost[1])
  
          diff = sorted(diff)
  
          return sums + sum(diff[:len(diff)//2])
  ```

### 116. 距离顺序排列矩阵单元格

* 索引排序

  ```python
  class Solution:
  
      def allCellsDistOrder(self, R: int, C: int, r0: int, c0: int) -> List[List[int]]:
  
          ans = []
          dist = []
          for r in range(R):
              for c in range(C):
                  d = abs(r-r0) + abs(c-c0)
                  ans.append([r, c])
                  dist.append(d)
  
          dists = list(zip(ans, dist))
          dists = sorted(dists, key=lambda d: d[-1])
          dists = [d[0] for d in dists]
  
          return dists
  
  ```

* 字典

  ```python
  class Solution:
  
      def allCellsDistOrder(self, R: int, C: int, r0: int, c0: int) -> List[List[int]]:
  
          dists = {}
          for r in range(R):
              for c in range(C):
                  d = abs(r-r0) + abs(c-c0)
                  if d in dists:
                      dists[d].append([r, c])
                  else:
                      dists[d] = [[r,c]]
  
          ans = []
  
          keys = sorted(dists.keys())
          for k in keys:
              ans.extend(dists[k])
  
          return ans
  ```

### 117. 移动石子直到连续

* 分情况讨论

  ```python
  class Solution:
      def numMovesStones(self, a: int, b: int, c: int) -> List[int]:
  
          seq = list(sorted([a, b, c]))
          if seq[-1]-seq[0]==2:
              return [0, 0]
          
          if seq[-1]-seq[1]==1:
              return [1, seq[1]-seq[0]-1]
          
          if seq[1]-seq[0]==1:
              return [1, seq[-1]-seq[1]-1]
  
          if seq[-1]-seq[1]==2:
              return [1, seq[-1]-seq[0]-2]
  
          if seq[1]-seq[0]==2:
              return [1, seq[-1]-seq[0]-2]
  
          return [2, seq[-1]-seq[0]-2]
  ```

* 分情况简化

  ```python
  class Solution:
      def numMovesStones(self, a: int, b: int, c: int) -> List[int]:
  
          seq = list(sorted([a, b, c]))
          if seq[-1]-seq[0]==2:
              return [0, 0]
  
          if seq[-1]-seq[1]<=2 or seq[1]-seq[0]<=2:
              return [1, seq[-1]-seq[0]-2]
          
          return [2, seq[-1]-seq[0]-2]
  ```

### 118. 有效的回旋镖

* 斜率

  ```python
  class Solution:
      def isBoomerang(self, points: List[List[int]]) -> bool:
  
          dx1, dy1 = points[1][0]-points[0][0], points[1][1]-points[0][1]
          dx2, dy2 = points[2][0]-points[0][0], points[2][1]-points[0][1]
  
          if dx1==0 and dy1==0 or dx2==0 and dy2==0:
              return False
  
          return dx1*dy2!=dx2*dy1
  ```

### 119. 查询后的偶数和

* 暴力

  ```python
  class Solution:
      def sumEvenAfterQueries(self, A: List[int], queries: List[List[int]]) -> List[int]:
  
          ans = []
          sums = sum(list(filter(lambda x: x%2==0, A)))
          for query in queries:
              val, index = query
              if A[index]%2!=0 and (A[index]+val)%2!=0:
                  A[index] += val
              elif A[index]%2==0 and (A[index]+val)%2!=0:
                  sums -= A[index]
                  A[index] += val
              elif A[index]%2!=0 and (A[index]+val)%2==0:
                  A[index] += val
                  sums += A[index]
              else:
                  sums += val
                  A[index] += val
              
              ans.append(sums)
  
          return ans
  ```

### 120. 词典中最长的单词

* 枚举

  ```python
  class Solution:
      def longestWord(self, words: List[str]) -> str:
  
          words = set(words)
  
          ans = ''
          for w in words:
              
              temp = w
              while w in words:
                  w = w[:-1]
              
              if len(w)==0:
                  if len(temp)==len(ans):
                      ans = min(ans, temp)
                  elif len(temp) > len(ans):
                      ans = temp
                  
          return ans
  ```

### 121. 寻找数组的中心索引

* 暴力

  ```python
  class Solution:
      def pivotIndex(self, nums: List[int]) -> int:
  
          sums = sum(nums)
  
          left = 0
          for i in range(len(nums)):
              if sums-left-nums[i]==left:
                  return i
              
              left += nums[i]
          
          return -1
  ```

### 122. 使用最小花费爬楼梯

* 动态规划

  ```python
  class Solution:
      def minCostClimbingStairs(self, cost: List[int]) -> int:
          
          if len(cost)==1:
              return cost[0]
          if len(cost)==2:
              return min(cost)
  
          opt = [0 for _ in range(len(cost)+1)]
          opt[0] = 0
          opt[1] = 0
          
          for i in range(2, len(cost)+1):
              opt1 = cost[i-1] + opt[i-1]
              opt2 = cost[i-2] + opt[i-2]
              opt[i] = min(opt1, opt2)
  
  
          return opt[-1]
  ```

  

### 123. 至少是其他数字两倍的最大数

* 暴力

  ```python
  class Solution:
      def dominantIndex(self, nums: List[int]) -> int:
      
          max_v = max(nums)
          idx = nums.index(max_v)
          for num in nums:
              if num!= max_v and 2*num > max_v:
                  return -1
  
          return idx
  ```

  

### 124. 寻找比目标字母大的最小字母

* 暴力

  ```python
  class Solution:
      def nextGreatestLetter(self, letters: List[str], target: str) -> str:
          
          if target >= letters[-1]:
              return letters[0]
  
          for c in letters:
              if target < c:
                  return c
  ```

### 125. 最短完整词

* 计数法

  ```python
  import sys
  class Solution:
      def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
  
          license = {}
          for x in licensePlate.lower():
              if not x.isalpha():
                  continue
  
              if x in license:
                  license[x] += 1
              else:
                  license[x] = 1
  
          min_v = sys.maxsize
          for word in words:
              count = {}
              for w in word.lower():
                  if w in count:
                      count[w] += 1
                  else:
                      count[w] = 1
  
              contains = True
              for k, v in license.items():
                  if k not in count or count[k] < v:
                      contains = False
                      break
              
              if contains and min_v==sys.maxsize:
                  ans = word
                  min_v = len(word)
              elif contains and min_v > len(word):
                  ans = word
                  min_v = len(word)
  
  
          return ans
  ```

  

### 126. 二进制表示中质数个计算置位

* 暴力法

  ```python
  class Solution:
  
      def isPrime(self, x):
          if x==0 or x==1:
              return False
          
          if x==2:
              return True
  
          i = 2
          while i**2 <= x:
              if x%i==0:
                  return False
              i += 1
  
          return True
  
  
      def countPrimeSetBits(self, L: int, R: int) -> int:
  
          count = 0
          for x in range(L, R+1):
              
              b = x
              a = 0
              while b:
                  a += b%2
                  b //= 2
  
              if self.isPrime(a):
                  count += 1
  
          return count
  ```

  

### 127. 托普利茨矩阵

* 暴力

  ```python
  class Solution:
      def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
  
          for t in range(len(matrix[0])):
              i = 0
              j = t
              while (i+1) < len(matrix) and (j+1) < len(matrix[0]):
                  if matrix[i][j]!=matrix[i+1][j+1]:
                      return False
                  i += 1
                  j += 1
          
          for t in range(len(matrix)):
              i = t
              j = 0
              while (i+1) < len(matrix) and (j+1) < len(matrix[0]):
                  if matrix[i][j]!=matrix[i+1][j+1]:
                      return False
                  i += 1
                  j += 1
  
          return True
  ```

### 128. 字母大小写全排列

* 递归

  ```python
  class Solution:
      def letterCasePermutation(self, S: str) -> List[str]:
  
          ans = ['']
          for i, s in enumerate(S):
              if s.isalpha():
                  temp = ans.copy()
                  ans = []
                  for t in temp:
                      ans.append(t + s.lower())
                  for t in temp:
                      ans.append(t + s.upper())
              else:
                  temp = ans.copy()
                  ans = []
                  for t in temp:
                      ans.append(t + s)
  
          return ans
  ```

### 129. 旋转数字

* 暴力

  ```python
  class Solution:
      def rotatedDigits(self, N: int) -> int:
  
          count = 0
          for x in range(1, N+1):
              b = str(x)
              if '3' in b or '4' in b or '7' in b:
                  continue
  
              if ('2' in b) or ('5' in b) or ('6' in b) or ('9' in b):
                  count += 1
  
          return count
              
  ```

### 130. 旋转字符串

* 暴力

  ```python
  class Solution:
      def rotateString(self, A: str, B: str) -> bool:
          if len(A)!=len(B):
              return False
          
          if len(A)==0:
              return True
  
          for _ in range(len(A)):
              if A==B:
                  return True
              else:
                  A = A[1:] + A[0]
  
          return False
  ```

### 131. 唯一的摩尔斯密码词

* 暴力

  ```python
  class Solution:
      def uniqueMorseRepresentations(self, words: List[str]) -> int:
          
          code = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
  
          ans = set()
          for word in words:
              
              mol = []
              for w in word:
                  idx = ord(w)-ord('a')
                  mol.append(code[idx])
  
              ans.add(''.join(mol))
  
          return len(ans)
  ```

### 132. 山羊拉丁文

* 暴力解法

  ```python
  class Solution:
      def toGoatLatin(self, S: str) -> str:
  
          if len(S) > 1:
              S = S.split(' ')
  
          for i in range(len(S)):
              s = S[i]
              if s[0].lower() in ('a', 'e', 'i', 'o', 'u'):
                  S[i] = s + 'ma' + 'a'*(i+1)
              else:
                  S[i] = s[1:] + s[0] + 'ma' + 'a'*(i+1)
  
          if len(S)==1:
              return S[0]
          else:
              return ' '.join(S)
  ```

### 133. 较大分组的位置

* 暴力

  ```python
  class Solution:
      def largeGroupPositions(self, S: str) -> List[List[int]]:
  
          ans = []
          
          i = 0
          while i < len(S)-1:
              
              while i < len(S)-1 and S[i]!=S[i+1]:
                  i += 1
              
              left = i
              while i < len(S)-1 and S[i]==S[i+1]:
                  i += 1
  
              if (i-left) >= 2:
                  ans.append([left, i])
  
          return ans
  ```

### 134. 字符的最短距离

* 暴力

  ```python
  class Solution:
      def shortestToChar(self, S: str, C: str) -> List[int]:
  
          index = []
          for i in range(len(S)):
              if S[i]==C:
                  index.append(i)
  
          ans = [0 for _ in range(len(S))]
  
          for j in range(0, index[0]):
              ans[j] = index[0]-j
          for j in range(index[-1], len(S)):
              ans[j] = j - index[-1]
  
          if len(index)==1:
              return ans
          
          for i in range(len(index)-1):
              for j in range(index[i], index[i+1]):
                  ans[j] = min(abs(index[i+1]-j), abs(j-index[i]))
  
          return ans
  ```


### 135. N皇后问题

* 回溯法（非递归）

  ```python
  def valid(nums, row):
      for i in range(row):
          if nums[i]==nums[row] or abs(row-i)==abs(nums[row]-nums[i]):
              return False
      return True
  
  def dfs(nums, ans):
      
      row = 0
      while True:
          
          while not valid(nums, row) and nums[row] < 4:
              nums[row] += 1
              
          
          if nums[row]==4 and row==0:
              break
          
          if nums[row]==4:
              nums[row] = 0
              row -= 1
              # 回退并推进一步
              nums[row] += 1
          else:
              row += 1
          
          if row == len(nums):
              ans.append(nums[:])
              row -= 1
              # 回退并推进一步
              nums[row] += 1
      
  
  nums = [0 for _ in range(4)]
  ans = []
  dfs(nums, ans)
  ```

  

* 回溯法（深度优先递归）

  ```python
  def valid(nums, row):
      for i in range(row):
          if abs(row-i)==abs(nums[row]-nums[i]) or nums[i]==nums[row]:
              return False
          
      return True
  
  def dfs(nums, row, ans):
      if row==4:
          ans.append(nums[:])
          return
      
      for col in range(4):
          nums[row] = col
          if valid(nums, row):
              print(nums)
              dfs(nums, row+1, ans)
  
  nums = [0 for _ in range(4)]
  ans = []
  dfs(nums, 0, ans)
  
  ```

### 136. 全排列问题

* 交换法(递归)

  ```python
  def swap(nums, i, ans):
      
      if i==len(nums):
          ans.append(nums[:])
          return
      
      
      for j in range(i, len(nums)):
          temp = nums[i]
          nums[i] = nums[j]
          nums[j] = temp
          
          swap(nums, i+1, ans)
          
          temp = nums[i]
          nums[i] = nums[j]
          nums[j] = temp
              
  nums = [i for i in range(4)]
  ans = []
  swap(nums, 0, ans)
  ```

* 回溯法（递归）

  ```python
  # 全排列
  
  def valid(nums, row):
      for i in range(row):
          if nums[i]==nums[row]:
              return False
      return True
  
  def dfs(nums, row, ans):
      if row==4:
          ans.append(nums[:])
          return
      
      for col in range(4):
          nums[row] = col
          if valid(nums, row):
              dfs(nums, row+1, ans)
              
  nums = [i for i in range(4)]
  ans = []
  dfs(nums, 0, ans)
  ```

* 字典序法（非递归）

  ```python
  def word(nums, ans):
      
      while True:
          
          ans.append(nums[:])
          
          i = len(nums)-1
          while i > 0 and nums[i] < nums[i-1]:
              i -= 1
          
          if i==0:
              break
              
          idx = i
          for j in range(i, len(nums)):
              if nums[j] > nums[i-1] and nums[idx] > nums[j]:
                  idx = j
          
          temp = nums[i-1]
          nums[i-1] = nums[idx]
          nums[idx] = temp
          
          nums[i:] = nums[:i-1:-1]
              
  nums = [1, 2, 3, 4]
  ans = []
  word(nums, ans)
  for a in ans:
      print(a)
  ```

* 回溯法（非递归）

  ```python
  def valid(nums, row):
      for i in range(row):
          if nums[row]==nums[i]:
              return False
      return True
  
  def dfs(nums, ans):
      
      row = 0
      while True:
          # 找到下一个有效状态
          while not valid(nums, row) and nums[row] < 4:
              nums[row] += 1
  		# 设置终止条件
          if nums[row]==4 and row==0:
              break
          # 设置回溯条件
          if nums[row]==4:
              nums[row] = 0
              row -= 1
              # 回退并推进一步
              nums[row] += 1
          else:
              row += 1
          # 设置出口
          if row == len(nums):
              ans.append(nums[:])
              row -= 1
              # 回退并推进一步
              nums[row] += 1
      
  
  nums = [1, 1, 1, 1]
  ans = []
  dfs(nums, ans)
  ```

### 137. 组合问题

* 回溯法（非递归）

  ```python
  def valid(nums, row):
      for i in range(row):
          if nums[row] <= nums[i]:
              return False
      return True
  
  def dfs(nums, ans):
      
      row = 0
      while True:
          
          while not valid(nums, row) and nums[row] < 6:
              nums[row] += 1
              
          
          if nums[row]==6 and row==0:
              break
          
          if nums[row]==6:
              nums[row] = 0
              row -= 1
              # 回退并推进一步
              nums[row] += 1
          else:
              row += 1
          
          if row == len(nums):
              ans.append(nums[:])
              row -= 1
              # 回退并推进一步
              nums[row] += 1
      
  
  nums = [0 for _ in range(4)]
  ans = []
  dfs(nums, ans)
  ```

* 回溯法（递归）

  ```python
  def valid(nums, row):
      for i in range(row):
          if nums[i] >= nums[row]:
              return False
      return True
  
  def dfs(nums, row, ans):
      if row==4:
          ans.append(nums[:])
          return
      
      for col in range(6):
          nums[row] = col
          if valid(nums, row):
              dfs(nums, row+1, ans)
              
  nums = [0 for _ in range(4)]
  ans = []
  dfs(nums, 0, ans)
  ```

  

### 138. 将0变成1的操作次数

* 依题意

  ```python
  class Solution:
      def numberOfSteps (self, num: int) -> int:
  
          ans = 0
          while num:
              if num%2==0:
                  num //= 2
              else:
                  num -= 1
              ans += 1
  
          return ans
  ```

### 139. 检查整数及其两倍数是否存在

* 哈希表

  ```python
  class Solution:
      def checkIfExist(self, arr: List[int]) -> bool:
  
          
          count = {}
          for i, x in enumerate(arr):
              count[x] = i
          for i, x in enumerate(arr):
              if 2*x in count and i!=count[2*x]:
                  return True
  
          return False
  ```



### 140. 和为零的N个唯一整数

* 暴力

  ```python
  class Solution:
      def sumZero(self, n: int) -> List[int]:
  
          ans = []
          for i in range(1, n//2+1):
              ans.append(i)
              ans.append(-i)
  
          if n%2!=0:
              ans.append(0)
  
          return ans
  ```

### 141. 不邻接植花

* 邻接表

  ```python
  class Solution:
  
      def valid(self, flowers, x, ys):
          for y in ys:
              if flowers[y-1]==flowers[x-1]:
                  return False
  
          return True
  
      def gardenNoAdj(self, N: int, paths: List[List[int]]) -> List[int]:
  
          graph = {}
          for path in paths:
              x, y = path
              if x in graph:
                  graph[x].append(y)
              else:
                  graph[x] = [y]
  
              if y in graph:
                  graph[y].append(x)
              else:
                  graph[y] = [x]
  
  
          flowers = [0 for _ in range(N)]
  
          for x in range(1, N+1):
              if not x in graph:
                  flowers[x-1] = 1
                  continue
  
              ys = graph[x]
              for i in range(1, N+1):
                  flowers[x-1] = i
                  if self.valid(flowers, x, ys):
                      break
              
          return flowers
  ```

### 142. 验证外星语词典

* 暴力

  ```python
  class Solution:
  
      def compare(self, a, b, index):
          if a==b:
              return False
  
          i = 0
          while i < len(a) and i < len(b) and a[i]==b[i]:
              i += 1
          
          if i < len(a) and i < len(b):
              return True if index[a[i]] > index[b[i]] else False
          elif i >= len(a):
              return False
          elif i >= len(b):
              return True
          
          
  
      def isAlienSorted(self, words: List[str], order: str) -> bool:
          index = {}
          for i, o in enumerate(order):
              index[o] = i
  
          for i in range(len(words)-1):
              if self.compare(words[i], words[i+1], index):
                  return False
          
          return True
  ```

### 143. 最大三角形面积

* 暴力

  ```python
  import math
  
  class Solution:
      def largestTriangleArea(self, points: List[List[int]]) -> float:
  
          max_area = 0
          for i in range(len(points)-2):
              for j in range(i+1, len(points)-1):
                  for k in range(j+1, len(points)):
                      x1, y1 = points[i]
                      x2, y2 = points[j]
                      x3, y3 = points[k]
                      va = [x2-x1, y2-y1]
                      vb = [x3-x1, y3-y1]
                      mod_va = (va[0]**2 + va[1]**2)**0.5
                      mod_vb = (vb[0]**2 + vb[1]**2)**0.5
                      cosine_2 = (va[0]*vb[0] + va[1]*vb[1])/(mod_va*mod_vb)
                      if cosine_2**2 > 1:
                          sine = 0
                      else:
                          sine = math.sqrt(1 - cosine_2**2)
                      area = abs(mod_va*mod_vb*sine/2)
                      max_area = max(max_area, area)
  
          return max_area
  ```

### 144. 二进制间距

* 暴力

  ```python
  class Solution:
      def binaryGap(self, N: int) -> int:
  
          ans = []
          count = 0
          while N:
              if N%2==1:
                  ans.append(count)
              N //= 2
  
              count += 1
          
          if len(ans) < 2:
              return 0
          
          max_v = 0
          for i in range(len(ans)-1):
              max_v = max(max_v, ans[i+1]-ans[i])
  
          return max_v
  ```



### 145. 模拟行走机器人

* 暴力

  ```python
  class Solution:
      def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
          
          obs = set()
          for (x, y) in obstacles:
              obs.add((x, y))
  
          d = 0
          x, y = 0, 0
          max_v = 0
          for cmd in commands:
              if cmd==-1:
                  d = (d+1)%4
              elif cmd==-2:
                  d = (d+3)%4
              else:
                  if d==1:
                      dx = 1
                      while dx <= cmd and (x+dx, y) not in obs:
                          dx += 1
                      x = x + dx - 1
  
                  if d==2:
                      dy = 1
                      while dy <= cmd and (x, y-dy) not in obs:
                          dy += 1
                      y = y - dy + 1
                  
                  if d==3:
                      dx = 1
                      while dx <= cmd and (x-dx, y) not in obs:
                          dx += 1
                      x = x - dx + 1
  
                  if d==0:
                      dy = 1
                      while dy <= cmd and (x, y+dy) not in obs:
                          dy += 1
                      y = y + dy - 1
  
  
                  max_v = max(max_v, x**2 + y**2)
  
  
          return max_v
  ```

### 146. 转置矩阵

* 暴力

  ```python
  class Solution:
      def transpose(self, A: List[List[int]]) -> List[List[int]]:
  
          ans = [[None for _ in range(len(A))] for _ in range(len(A[0]))]
  
          for i in range(len(A)):
              for j in range(len(A[0])):
                  ans[j][i] = A[i][j]
  
          return ans
  ```

### 147. 柠檬水找零

* 暴力

  ```python
  class Solution:
      def lemonadeChange(self, bills: List[int]) -> bool:
  
          money = []
  
          for b in bills:
              if b==5:
                  money.insert(0, 5)
              elif b==10:
                  money.append(b)
                  if money[0]!=5:
                      return False
                  money.pop(0)
              elif b==20:
                  if len(money) >= 2 and money[0]==5 and money[-1]==10:
                      money.pop(0)
                      money.pop()
                  elif len(money) >= 3 and money[0]==5 and money[1]==5 and money[2]==5:
                      money.pop(0)
                      money.pop(0)
                      money.pop(0)
                  else:
                      return False
                  
          return True
  ```

  

### 148. 矩形重叠

* 暴力

  ```python
  class Solution:
      def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
  
          rx = [rec1[0], rec1[2], rec2[0], rec2[2]]
          ry = [rec1[1], rec1[3], rec2[1], rec2[3]]
  
          rw = [abs(rec1[2]-rec1[0]), abs(rec2[2]-rec2[0])]
          rh = [abs(rec1[3]-rec1[1]), abs(rec2[3]-rec2[1])]
  
          if max(rx)-min(rx) < sum(rw) and max(ry)-min(ry) < sum(rh):
              return True
          
          return False
  ```

* 检查位置

  ```python
  class Solution:
      def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
  
          if rec1[2] <= rec2[0] or rec1[0] >= rec2[2] \
          or rec1[1] >= rec2[3] or rec1[3] <= rec2[1]:
              return False
          
          return True
  ```

### 149. 写字符串需要的行数

* 一遍扫描

  ```python
  class Solution:
      def numberOfLines(self, widths: List[int], S: str) -> List[int]:
  
          row = 0
          sums = 0
          for s in S:
              idx = ord(s)-ord('a')
              if sums + widths[idx] <= 100:
                  sums += widths[idx]
              else:
                  row += 1
                  sums = widths[idx]
  
          row += 1 if sums > 0 else 0
          return [row, sums]
  ```

### 150. 环形链表

* 标记法

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def hasCycle(self, head: ListNode) -> bool:
  
          p = head
          while p:
              if p.val==sys.maxsize:
                  return True
              else:
                  p.val = sys.maxsize
                  p = p.next
  
          return False
  ```

* 集合

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def hasCycle(self, head: ListNode) -> bool:
  
          visit = set()
          p = head
          while p and id(p) not in visit:
              visit.add(id(p))
              p = p.next
          
          if p==None:
              return False
          else:
              return True
  ```

* 快慢指针

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def hasCycle(self, head: ListNode) -> bool:
  
          slow = head
          fast = head
          while slow and fast:
              slow = slow.next
              
              fast = fast.next
              if fast:
                  fast = fast.next
              else:
                  return False
  
              if id(slow)==id(fast):
                  return True
          
          return False
  ```

### 151. 相交链表

* 集合

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
  
          p = headA
          visit = set()
          while p:
              visit.add(id(p))
              p = p.next
  
          p = headB
          while p and id(p) not in visit:
              p = p.next
  
          return p
  ```

* 双指针

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
  
          p = headA
          q = headB
          
          count = 0
          while p!=q:
              if p and p.next:
                  p = p.next
              else:
                  count += 1
                  p = headB
              
              if q and q.next:
                  q = q.next
              else:
                  q = headA
  
              if p==q:
                  return p
              if count >= 2:
                  return None
  
          if p==q:
              return p
          else:
              return None
  ```

* 双指针2

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
  
          p = headA
          q = headB
          if p==q:
              return p
  
          while p!=q:
              p = p.next if p else headB
              q = q.next if q else headA
  
              if p==headA and q==headB:
                  return None
          
          return p
  ```

  

### 152. 两个数组的交集II

* 计数

  ```python
  class Solution:
      def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
  
          counts0 = {}
          for num in nums1:
              if num in counts0:
                  counts0[num] += 1
              else:
                  counts0[num] = 1
              
          counts1 = {}
          for num in nums2:
              if num in counts1:
                  counts1[num] += 1
              else:
                  counts1[num] = 1
  
          counts = {}
          for k, v in counts0.items():
              if k in counts1:
                  counts[k] = min(counts1[k], v)
          
          for k, v in counts1.items():
              if k in counts0:
                  counts[k] = min(counts0[k], v)
  
          ans = []
          for k, v in counts.items():
              for _ in range(v):
                  ans.append(k)
  
          return ans
  ```

* 排序

  ```python
  class Solution:
      def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
  
          nums1.sort()
          nums2.sort()
  
          i = 0
          j = 0
          ans = []
          while i < len(nums1) and j < len(nums2):
              if nums1[i] < nums2[j]:
                  i += 1
              elif nums1[i] > nums2[j]:
                  j += 1
              else:
                  ans.append(nums1[i])
                  i += 1
                  j += 1
          
          return ans
  ```

* 计数2

  ```python
  class Solution:
      def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
  
          
          counts = {}
          for num in nums1:
              if num in counts:
                  counts[num] += 1
              else:
                  counts[num] = 1
  
          ans = []
          for num in nums2:
              if num in counts and counts[num] > 0:
                  ans.append(num)
                  counts[num] -= 1
  
          return ans
  ```

### 153. 打家劫舍

* 动态规划

  ```python
  class Solution:
      def rob(self, nums: List[int]) -> int:
  
          if len(nums)==0:
              return 0
          if len(nums)==1:
              return nums[0]
          if len(nums)==2:
              return max(nums)
          
          opt = [0 for _ in range(len(nums))]
          opt[0] = nums[0]
          opt[1] = max(nums[0], nums[1])
          for i in range(2, len(nums)):
              opt1 = nums[i] + opt[i-2]
              opt2 = opt[i-1]
              opt[i] = max(opt1, opt2)
  
          return opt[-1]
  ```

* 动态规划（空间）

  ```python
  class Solution:
      def rob(self, nums: List[int]) -> int:
  
          if len(nums)==0:
              return 0
          if len(nums)==1:
              return nums[0]
          if len(nums)==2:
              return max(nums)
          
          opt = [0 for _ in range(len(nums))]
          opt[0] = nums[0]
          opt[1] = max(nums[0], nums[1])
          for i in range(2, len(nums)):
              opt1 = nums[i] + opt[i-2]
              opt2 = opt[i-1]
              opt[i] = max(opt1, opt2)
  
          return opt[-1]
  ```

### 154. 移除链表元素

* 暴力

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def removeElements(self, head: ListNode, val: int) -> ListNode:
  
          p = head
          while p and p.val==val:
              p = p.next
          
          if p==None:
              return p
  
          head = p
          q = p.next
          while q:
              if q.val==val:
                  p.next = q.next
                  q = p.next
              else:
                  p = p.next
                  q = q.next
  
          return head
  ```

### 155. 用队列实现栈

* 双队列（push O(n)， pop O(1))

  ```python
  import queue
  
  class MyStack:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.q1 = queue.Queue()
          self.q2 = queue.Queue()
          
  
      def push(self, x: int) -> None:
          """
          Push element x onto stack.
          """
          while not self.q1.empty():
              self.q2.put(self.q1.get())
          self.q1.put(x)
  
          while not self.q2.empty():
              self.q1.put(self.q2.get())
  
          
  
      def pop(self) -> int:
          """
          Removes the element on top of the stack and returns that element.
          """
  
          return self.q1.get()
          
  
      def top(self) -> int:
          """
          Get the top element.
          """
          top = self.q1.get()
          self.push(top)
          return top
  
  
      def empty(self) -> bool:
          """
          Returns whether the stack is empty.
          """
  
          return self.q1.empty()
  ```

* 单队列

  ```python
  import queue
  
  class MyStack:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.q = queue.Queue()
          
  
      def push(self, x: int) -> None:
          """
          Push element x onto stack.
          """
          self.q.put(x)
          for _ in range(self.q.qsize()-1):
              self.q.put(self.q.get())
          
  
      def pop(self) -> int:
          """
          Removes the element on top of the stack and returns that element.
          """
  
          return self.q.get()
          
  
      def top(self) -> int:
          """
          Get the top element.
          """
          top = self.q.get()
          self.push(top)
          return top
  
  
      def empty(self) -> bool:
          """
          Returns whether the stack is empty.
          """
  
          return self.q.empty()
  ```

### 156. 翻转二叉树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def invert(self, root):
          
          if root==None:
              return
  
          left, right = root.left, root.right
          root.right = left
          root.left = right
  
          self.invertTree(root.left)
          self.invertTree(root.right)
          
  
      def invertTree(self, root: TreeNode) -> TreeNode:
  
          self.invert(root)
          return root
  ```

* 队列

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import queue
  class Solution:
  
      def invertTree(self, root: TreeNode) -> TreeNode:
  
          q = queue.Queue()
          q.put(root)
  
          while not q.empty():
              top = q.get()
              if top==None:
                  continue
              q.put(top.left)
              q.put(top.right)
  
              left = top.left
              right = top.right
              top.left = right
              top.right = left
          
          return root
  ```

### 157. 二叉树所有路径

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def binaryPaths(self, root, path, paths):
  
          if path=='':
              path = str(root.val)
          else:
              path = path + '->' + str(root.val)
  
          if root.left:
              self.binaryPaths(root.left, path, paths)
          if root.right:
              self.binaryPaths(root.right, path, paths)
  
          if root.left==None and root.right==None:
              paths.append(path)
  
          
  
      def binaryTreePaths(self, root: TreeNode) -> List[str]:
          if root==None:
              return []
  
          ans = []
          self.binaryPaths(root, '', ans)
          return ans
  ```

* 递归(列表 )

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def binaryPaths(self, root, path, paths):
  
          path.append(root.val)
          if root.left:
              self.binaryPaths(root.left, path[:], paths)
          if root.right:
              self.binaryPaths(root.right, path[:], paths)
  
          if root.left==None and root.right==None:
              paths.append(path[:])
          
  
      def binaryTreePaths(self, root: TreeNode) -> List[str]:
          if root==None:
              return []
  
          paths = []
          self.binaryPaths(root, [], paths)
          ans = []
          for path in paths:
              ans.append('->'.join(list(map(str, path))))
  
          return ans
  ```


### 158. 二叉搜索树的最近公共祖先

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def CommonAncestor(self, root, p, q):
  
          if p.val > root.val and q.val > root.val:
              return self.CommonAncestor(root.right, p, q)
          elif p.val < root.val and q.val < root.val:
              return self.CommonAncestor(root.left, p, q)
          else:
              return root
          
  
  
      def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
  
          return self.CommonAncestor(root, p, q)
  ```

* 迭代法

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
  
          while True:
              
              if p.val > root.val and q.val > root.val:
                  root = root.right
              elif p.val < root.val and q.val < root.val:
                  root = root.left
              else:
                  return root
  ```

### 159. 回文链表

* 列表

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def isPalindrome(self, head: ListNode) -> bool:
  
          ans = []
          
          p = head
          while p:
              ans.append(p.val)
              p = p.next
          
          i = 0
          j = len(ans)-1
          while i < j and ans[i]==ans[j]:
              i += 1
              j -= 1
          
          if i >= j:
              return True
          else:
              return False
  ```


### 160. 猜数字大小

* 二分法

  ```python
  # The guess API is already defined for you.
  # @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
  # def guess(num: int) -> int:
  
  class Solution:
      def guessNumber(self, n: int) -> int:
  
          left = 1
          right = n
          mid = (left + right)//2
          while left <= right:
              if guess(mid)==0:
                  return mid
              elif guess(mid)==1:
                  left = mid + 1
              else:
                  right = mid - 1
  
              mid = (left + right)//2
  ```

### 161. 赎金信

* 计数

  ```python
  class Solution:
      def canConstruct(self, ransomNote: str, magazine: str) -> bool:
  
          counts = {}
          for m in magazine:
              if m in counts:
                  counts[m] += 1
              else:
                  counts[m] = 1
  
          for r in ransomNote:
              if r not in counts:
                  return False
              elif counts[r] > 0:
                  counts[r] -= 1
              else:
                  return False
              
          return True
  ```

### 162. 字符串中的第一个唯一字符

* 计数

  ```python
  class Solution:
      def firstUniqChar(self, s: str) -> int:
  
          counts = {}
          for x in s:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
              
          for i, x in enumerate(s):
              if counts[x]==1:
                  return i
              
          return -1
  ```

### 163. 找不同

* 计数法

  ```python
  class Solution:
      def findTheDifference(self, s: str, t: str) -> str:
          
          count_s = {}
          count_t = {}
          for x in s:
              if x in count_s:
                  count_s[x] += 1
              else:
                  count_s[x] = 1
          
          for x in t:
              if x in count_t:
                  count_t[x] += 1
              else:
                  count_t[x] = 1
  
          for k, v in count_t.items():
              if k not in count_s or count_t[k] > count_s[k]:
                  return k
  ```

* 排序

  ```python
  class Solution:
      def findTheDifference(self, s: str, t: str) -> str:
          
          s = list(s)
          t = list(t)
          s.sort()
          t.sort()
  
          i = 0
          while i < min(len(s), len(t)) and s[i]==t[i]:
              i += 1
          
          if i < min(len(s), len(t)):
              return t[i]
          else:
              return t[-1]
  ```

### 164. 判断子序列

* 扫描

  ```python
  class Solution:
      def isSubsequence(self, s: str, t: str) -> bool:
  
          i = 0
          for x in s:
              
              while i < len(t) and t[i]!=x:
                  i += 1
              
              if i < len(t):
                  i += 1
              else:
                  return False
  
          return True
  ```

### 165. 二进制手表

* 暴力

  ```python
  class Solution:
  
      def count(self, v):
          sums = 0
          while v:
              sums += v%2
              v //= 2
          return sums
  
      def readBinaryWatch(self, num: int) -> List[str]:
  
          ans = []
  
          for h in range(12):
              ch = self.count(h)
              for m in range(60):
                  cm = self.count(m)
                  if (ch+cm)!=num:
                      continue
  
                  if m < 10:
                      ans.append('{}:0{}'.format(h, m))
                  else:
                      ans.append('{}:{}'.format(h, m)) 
  
          return ans
  ```

### 166. 左叶子之和

* 深度优先遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, ans):
          if root==None:
              return
          
          if root.left and root.left.left==None and root.left.right==None:
              ans.append(root.left.val)
          
          if root.right:
              self.dfs(root.right, ans)
  
          if root.left:
              self.dfs(root.left, ans)
  
              
  
      def sumOfLeftLeaves(self, root: TreeNode) -> int:
  
          ans = []
          self.dfs(root, ans)
  
          return sum(ans)
  ```

* 队列

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  import queue
  class Solution:
  
      def sumOfLeftLeaves(self, root: TreeNode) -> int:
          if root==None:
              return 0
  
          q = queue.Queue()
          q.put(root)
  
          sums = 0
          while not q.empty():
              top = q.get()
              if top.right:
                  q.put(top.right)
              if top.left and top.left.left==None and top.left.right==None:
                  sums += top.left.val
              elif top.left:
                  q.put(top.left)
              
          return sums
  ```

* 栈

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def sumOfLeftLeaves(self, root: TreeNode) -> int:
          if root==None:
              return 0
  
          stack = []
          stack.append(root)
  
          sums = 0
          while len(stack) > 0:
              top = stack.pop()
              if top.right:
                  stack.append(top.right)
              if top.left and top.left.left==None and top.left.right==None:
                  sums += top.left.val
              elif top.left:
                 stack.append(top.left)
              
          return sums
  ```

### 167. 数字转换为十六进制数

* 暴力

  ```python
  class Solution:
      def toHex(self, num: int) -> str:
  
          if num==0:
              return "0"
  
          if num < 0:
              num = 2**32+num
          
          ans = []
          while num:
              ans.append(num%16)
              num //= 16
          
          ans = [str(x) if x < 10 else chr(ord('a') + x - 10) for x in ans]
  
          return ''.join(ans[::-1])
          
  ```


### 168. 最长回文串

* 计数

  ```python
  class Solution:
      def longestPalindrome(self, s: str) -> int:
  
          counts = {}
          for x in s:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
          
          ans = 0
          for k, v in counts.items():
              if v%2==0:
                  ans += v
              else:
                  ans += v-1
  
          return ans if ans==len(s) else ans+1
  ```

### 169. Fizz Buzz

* 暴力

  ```python
  class Solution:
      def fizzBuzz(self, n: int) -> List[str]:
  
          ans = []
          for i in range(1, n+1):
              if i%3==0 and i%5==0:
                  ans.append('FizzBuzz')
              elif i%5==0:
                  ans.append('Buzz')
              elif i%3==0:
                  ans.append('Fizz')
              else:
                  ans.append(str(i))
  
          return ans
  ```

### 170. 第三大的数

* 三次遍历

  ```python
  class Solution:
      def thirdMax(self, nums: List[int]) -> int:
  
          maxes = []
          for _ in range(3):
              max_v = -sys.maxsize
  
              for num in nums:
                  if num not in maxes:
                      max_v = max(num, max_v)
  
              if max_v==-sys.maxsize:
                  continue
  
              maxes.append(max_v)
  
          if len(maxes)==3:
              return maxes[-1]
          else:
              return maxes[0]
  ```

### 172. 字符串相加

* 暴力

  ```python
  class Solution:
      def addStrings(self, num1: str, num2: str) -> str:
  
          ans = []
          num1 = list(map(int, list(reversed(list(num1)))))
          num2 = list(map(int, list(reversed(list(num2)))))
  
          i = 0
          pour = 0
          while i < len(num1) and i < len(num2):
              ans.append((num1[i] + num2[i] + pour)%10)
              pour = (num1[i] + num2[i] + pour)//10
              i += 1
          
          while i < len(num1):
              ans.append((num1[i] + pour)%10)
              pour = (num1[i] + pour)//10
              i += 1
  
          while i < len(num2):
              ans.append((num2[i] + pour)%10)
              pour = (num2[i] + pour)//10
              i += 1
  
          if pour==1:
              ans.append(pour)
          ans = ''.join(list(map(str, list(reversed(ans)))))
  
          return ans
  
  ```

### 173. 字符串中单词的个数

* 暴力

  ```python
  class Solution:
      def countSegments(self, s: str) -> int:
  
          i = 0
          count = 0
  
          while i < len(s) and s[i]==' ':
              i += 1
  
          while i < len(s):
              
              while i < len(s) and s[i]!=' ':
                  i += 1
              
              count += 1
              
              while i < len(s) and s[i]==' ':
                  i += 1
          
          return count
  ```



### 174. 排列硬币

* 暴力法

  ```python
  class Solution:
      def arrangeCoins(self, n: int) -> int:
  
          i = 1
          while n >= i:
              n -= i
              i += 1
          
          return i-1
  ```

* 公式

  ```python
  class Solution:
      def arrangeCoins(self, n: int) -> int:
  
          return int((math.sqrt(1+8*n)-1)/2)
  ```

### 175. 压缩字符串

* 暴力

  ```python
  class Solution:
      def compress(self, chars: List[str]) -> int:
  
          i = 0
          j = 0
  
          while j < len(chars):
              
              count = 0
              while j < len(chars) and chars[i]==chars[j]:
                  j += 1
                  count += 1
  
              if j < len(chars) and count==1:
                  i += 1
                  chars[i] = chars[j]
              elif j < len(chars) and count > 1:
                  count = str(count)
                  i += 1
                  for c in count:
                      chars[i] = c
                      i += 1
                  chars[i] = chars[j]
              elif j >= len(chars) and count==1:
                  return i + 1
              elif j >= len(chars) and count > 1:
                  count = str(count)
                  i += 1
                  for c in count:
                      chars[i] = c
                      i += 1
                  return i
  
              
  
              count = 0
  ```

### 176. 回旋镖的数量

* 计数法

  ```python
  class Solution:
      def numberOfBoomerangs(self, points: List[List[int]]) -> int:
  
          ans = 0
          for i in range(len(points)):
              dist = {}
  
              for j in range(len(points)):
                  d = (points[j][0]-points[i][0])**2 + \
                  (points[j][1]-points[i][1])**2
                  if d in dist:
                      dist[d] += 1
                  else:
                      dist[d] = 1
                  
              for v in dist.values():
                  ans += v*(v-1)
              
          return ans
  ```

### 177. 重复的子字符串

* 暴力

  ```python
  class Solution:
      def repeatedSubstringPattern(self, s: str) -> bool:
  
          if len(s)==1:
              return False
  
          i = 1
          while i < len(s):
  
              while i < len(s) and s[i]!=s[0]:
                  i += 1
              
              if i < len(s) and len(s)%i==0:
                  n = len(s)//i
                  base = s[:i]
                  if n > 1 and s==base*n:
                      return True
  
              i += 1
  
          return False
  ```

### 178. 找出数组中所有消失的数字

* 桶排序

  ```python
  class Solution:
      def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
  
          i = 0
          while i < len(nums):
              if nums[i]==i+1 or nums[nums[i]-1]==nums[i]:
                  i += 1
              else:
                  temp = nums[i]
                  nums[i] = nums[temp-1]
                  nums[temp-1] = temp
          
          ans = []
          for i in range(1, len(nums) + 1):
              if nums[i-1]!=i:
                  ans.append(i)
              
          return ans
  ```


### 179. 岛屿的周长

* 暴力

  ```python
  class Solution:
  
      def circle(self, grid, i, j):
          c = 0
          if i==0 or grid[i-1][j]==0:
              c += 1
          if i==len(grid)-1 or grid[i+1][j]==0:
              c += 1
          if j==0 or grid[i][j-1]==0:
              c += 1
          if j==len(grid[0])-1 or grid[i][j+1]==0:
              c += 1
  
          return c
  
      def islandPerimeter(self, grid: List[List[int]]) -> int:
  
          sums = 0
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j]==1:
                      sums += self.circle(grid, i, j)
          
          return sums
  ```

* 不使用函数封装，速度更快

  ```python
  class Solution:
  
  
      def islandPerimeter(self, grid: List[List[int]]) -> int:
  
          sums = 0
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j]==1:
                      c = 0
                      if i==0 or grid[i-1][j]==0:
                          c += 1
                      if i==len(grid)-1 or grid[i+1][j]==0:
                          c += 1
                      if j==0 or grid[i][j-1]==0:
                          c += 1
                      if j==len(grid[0])-1 or grid[i][j+1]==0:
                          c += 1
                      sums += c
          
          return sums
  ```

### 180. 数字的补数

* 列表

  ```python
  class Solution:
      def findComplement(self, num: int) -> int:
  
          v = []
          while num:
              v.append(1 - num%2)
              num //= 2
  
          v = list(reversed(v))
  
          ans = 0
          for x in v:
              ans = ans*2 + x
  
          return ans
  ```

* 移位

  ```python
  class Solution:
      def findComplement(self, num: int) -> int:
  
          temp = 1
          while temp < num:
              temp = temp*2 + 1
  
          return temp - num
  ```

### 181. 斐波那契数

* 动态规划

  ```python
  class Solution:
      def fib(self, N: int) -> int:
  
          if N <= 1:
              return N
  
          f0 = 0
          f1 = 1
          for _ in range(2, N+1):
              temp = f1
              f1 += f0
              f0 = temp
          
          return f1
  ```

### 182. 七进制数

* 暴力

  ```python
  class Solution:
      def convertToBase7(self, num: int) -> str:
  
          sign = 1 if num >= 0 else -1
          num = abs(num)
  
          ans = []
          while num >= 7:
              ans.append(num%7)
              num = num//7
  
          ans.append(num)
  
          ans = list(reversed(ans))
          ans = list(map(str, ans))
  
          if sign==1:
              return ''.join(ans)
          else:
              return '-' + ''.join(ans)
  ```

### 183. 密钥格式化

* 暴力

  ```python
  class Solution:
      def licenseKeyFormatting(self, S: str, K: int) -> str:
  
          ans = []
          for s in S:
              if s=='-':
                  continue
  
              ans.append(s.upper())
  
          first = len(ans)%K
          i = 0
          s = []        
          while i < len(ans):
              if i%K==first and i!=0:
                  s.append('-')
  
              s.append(ans[i])
              i += 1
  
          return ''.join(s)
  ```

### 184. 二叉搜索树中的众数

* 暴力

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      
      def dfs(self, root, counts):
          if root==None:
              return
          
          if root.val in counts:
              counts[root.val] += 1
          else:
              counts[root.val] = 1
          
          if root.left:
              self.dfs(root.left, counts)
          if root.right:
              self.dfs(root.right, counts)
  
  
      def findMode(self, root: TreeNode) -> List[int]:
  
          counts = {}
          self.dfs(root, counts)
  
          max_v = 0
          ans = []
          for k, v in counts.items():
              if v > max_v:
                  ans.clear()
                  ans.append(k)
                  max_v = v
  
              elif v==max_v:
                  ans.append(k)
          return ans
  ```

### 185. 键盘的行

* 集合

  ```python
  class Solution:
      def findWords(self, words: List[str]) -> List[str]:
  
          row1 = set(list('QWERTYUIOP'))
          row2 = set(list('ASDFGHJKL'))
          row3 = set(list('ZXCVBNM'))
          kbd = [row1, row2, row3]
  
          index = 0
          ans = []
          for word in words:
              temp = word
              word = word.upper()
              i = 0
              while i < len(kbd) and word[0] not in kbd[i]:
                  i += 1
  
              j = 0
              while j < len(word) and word[j] in kbd[i]:
                  j += 1
              
              if j >= len(word):
                  ans.append(temp)
  
          return ans
  ```

### 186. 下一个更大的元素

* 哈希

  ```python
  class Solution:
      def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
          
          nums11 = set(nums1)
  
          next_max = {}
  
          for i in range(len(nums2)):
              if nums2[i] not in nums11:
                  continue
              
              j = i
              while j < len(nums2) and nums2[j] <= nums2[i]:
                  j += 1
              
              if j >= len(nums2):
                  next_max[nums2[i]] = -1
              else:
                  next_max[nums2[i]] = nums2[j]
  
          return [next_max[x] for x in nums1]
  ```

* 单调栈

  ```python
  class Solution:
      def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
          
          next_max = {}
  
          stack = []
  
          for i in range(len(nums2)):
              if len(stack)==0:
                  stack.append(nums2[i])
              elif stack[-1] < nums2[i]:
                  while len(stack) > 0 and stack[-1] < nums2[i]:
                      next_max[stack.pop()] = nums2[i]
                  stack.append(nums2[i])
              else:
                  stack.append(nums2[i])
          
          ans = []
          for num in nums1:
              if num in next_max:
                  ans.append(next_max[num])
              else:
                  ans.append(-1)
  
          return ans
  ```

### 187. 最长特殊序列

* 简单解法

  ```python
  class Solution:
      def findLUSlength(self, a: str, b: str) -> int:
  
          if a==b:
              return -1
          elif len(a)==len(b):
              return len(a)
          else:
              return max(len(a), len(b))
  ```

### 188. 二叉搜索树的最小绝对差

* 中序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, ans):
          if root==None:
              return
          
          if root.left:
              self.dfs(root.left, ans)
          
          ans.append(root.val)
      
          if root.right:
              self.dfs(root.right, ans)
  
  
  
      def getMinimumDifference(self, root: TreeNode) -> int:
          
          ans = []
          self.dfs(root, ans)
          min_v = sys.maxsize
          for i in range(len(ans)-1):
              if ans[i+1]-ans[i] < min_v:
                  min_v = ans[i+1]-ans[i]
  
          return min_v
  ```

* 中序遍历2

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return
          
          if root.left:
              self.dfs(root.left)
          
          if self.prev==-1:
              self.prev = root.val
          else:
              self.min_v = min(self.min_v, root.val-self.prev)
              self.prev = root.val
      
          if root.right:
              self.dfs(root.right)
  
  
  
      def getMinimumDifference(self, root: TreeNode) -> int:
          
          self.min_v = sys.maxsize
          self.prev = -1
          self.dfs(root)
          return self.min_v
  ```


### 189. 反转字符串II

* 暴力

  ```python
  class Solution:
      def reverseStr(self, s: str, k: int) -> str:
  
  
          s = list(s)
  
          for i in range(0, len(s), 2*k):
              s[i:i+k] = reversed(s[i:i+k])
  
          return ''.join(s)       
  ```

### 190. 数组中的K-diff对

* 计数法

  ```python
  class Solution:
      def findPairs(self, nums: List[int], k: int) -> int:
  
          if k < 0:
              return 0
  
          counts = {}
          for num in nums:
              if num in counts:
                  counts[num] += 1
              else:
                  counts[num] = 1
  
          ans = 0
          if k==0:
              for num, v in counts.items():
                  if v > 1:
                      ans += 1
              
              return ans
          else:
              for num, v in counts.items():
                  if (num+k) in counts:
                      ans += 1
              
          return ans
  ```

### 191. 反转字符串中的单词 III

* 分割逆序

  ```python
  class Solution:
      def reverseWords(self, s: str) -> str:
  
          s = s.split(' ')
          ans = []
          for x in s:
              ans.append(x[::-1])
          
          return ' '.join(ans)
  ```

* 列表

  ```python
  class Solution:
      def reverseWords(self, s: str) -> str:
  
          s = list(s)
          
          j = 0
          while j < len(s):
              i = j
              while j < len(s) and s[j]!=' ':
                  j += 1
              
              s[i:j] = reversed(s[i:j])
              j += 1
  
          return ''.join(s)
  ```

### 192. 反转字符串中的元音字母

* 双指针

  ```python
  class Solution:
      def reverseVowels(self, s: str) -> str:
  
          yy = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
          s = list(s)
  
          i = 0
          j = len(s)-1
          while i < j:
              
              while i < j and s[i] not in yy:
                  i += 1
              
              while i < j and s[j] not in yy:
                  j -= 1
              
              if i < j:
                  temp = s[i]
                  s[i] = s[j]
                  s[j] = temp
                  i += 1
                  j -= 1
  
          return ''.join(s)
  ```

### 193. 猜数字游戏

* 计数法

  ```python
  class Solution:
      def getHint(self, secret: str, guess: str) -> str:
  
          scount = {}
          gcount = {}
  
          a = 0
          for i in range(len(secret)):
              if secret[i]==guess[i]:
                  a += 1
              else:
                  if secret[i] in scount:
                      scount[secret[i]] += 1
                  else:
                      scount[secret[i]] = 1
                  
                  if guess[i] in gcount:
                      gcount[guess[i]] += 1
                  else:
                      gcount[guess[i]] = 1
  
          b = 0
          for k, v in scount.items():
              if k in gcount:
                  b += min(v, gcount[k])
  
          return '{}A{}B'.format(a, b)
  ```

* 桶

  ```python
  class Solution:
      def getHint(self, secret: str, guess: str) -> str:
  
          sc = [0 for _ in range(10)]
          gc = [0 for _ in range(10)]
  
          a = 0
          for i in range(len(secret)):
              if secret[i]==guess[i]:
                  a += 1
                  continue
              sc[int(secret[i])] += 1
              gc[int(guess[i])] += 1
  
          b = 0
          for i in range(len(sc)):
              b += min(sc[i], gc[i])
          
          return '{}A{}B'.format(a, b)
  ```

### 194. 区域和检索 - 数组不可变

* 累加

  ```python
  class NumArray:
  
      def __init__(self, nums: List[int]):
          self.sums = nums[:]
          
          for i in range(1, len(nums)):
              self.sums[i] += self.sums[i-1]
          
      def sumRange(self, i: int, j: int) -> int:
          
          if i==0:
              return self.sums[j]
          else:
              return self.sums[j]-self.sums[i-1]
  ```

### 195. 平衡二叉树

* 从上到下计算深度

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def height(self, root):
          if root==None:
              return 0
          
          return 1 + max(self.height(root.left), self.height(root.right))
  
      def isBalanced(self, root: TreeNode) -> bool:
  
          if root==None:
              return True
          elif abs(self.height(root.left)-self.height(root.right)) > 1:
              return False
          else:
              return self.isBalanced(root.left) and self.isBalanced(root.right)
  ```

### 196. 构造矩形

* 暴力

  ```python
  class Solution:
      def constructRectangle(self, area: int) -> List[int]:
  
          W = int(math.sqrt(area))
  
          while area%W!=0:
              W -= 1
          
          return [area//W, W]
  ```

### 197. 供暖器

* 排序+二分

  ```python
  class Solution:
      def findRadius(self, houses: List[int], heaters: List[int]) -> int:
  
          houses.sort()
          heaters.sort()
  
          ans = 0
          for hs in houses:
  
              dist = None
              left = 0
              right = len(heaters)-1
              while left <= right:
                  mid = (left + right)//2
                  if heaters[mid]==hs:
                      dist = 0
                      break
                  
                  if heaters[mid] > hs:
                      right = mid - 1
                  else:
                      left = mid + 1
  
              if left > right:
                  if left < len(heaters) and right >= 0:
                      dist = min(heaters[left]-hs, hs-heaters[right])
                  elif left >= len(heaters) and right >=0:
                      dist = hs-heaters[right]
                  elif left < len(heaters) and right < 0:
                      dist = heaters[left]-hs
  
              
              ans = max(ans, dist)
  
          return ans
  ```



### 198. 亲密字符串

* 暴力

  ```python
  class Solution:
      def buddyStrings(self, A: str, B: str) -> bool:
  
          if len(A)!=len(B):
              return False
  
          if A==B:
              A = list(A)
              return len(set(A)) < len(A)
          
          diff = []
          for i in range(len(A)):
              if A[i]!=B[i]:
                  diff.append(i)
  
          if len(diff) != 2:
              return False
          
          i, j = diff
          
          return A[i]==B[j] and A[j] == B[i]
  ```


### 199. 最常见的单词

* 暴力解法

  ```python
  import re
  class Solution:
      def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
  
          counts = {}
          paragraph = paragraph.lower()
          banned = set(banned)
          
          i = 0
          while i < len(paragraph):
              
              j = i
              while i < len(paragraph) and paragraph[i].isalpha():
                  i += 1
  
              word = paragraph[j:i]
              if word not in banned:
                  if word in counts:
                      counts[word] += 1
                  else:
                      counts[word] = 1
  
              while i < len(paragraph) and not paragraph[i].isalpha():
                  i += 1
          
          max_v = 0
          idx = None
          for k, v in counts.items():
              if v > max_v:
                  max_v = v
                  idx = k
  
          return idx
  ```

* 分割

  ```python
  import re
  class Solution:
      def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
  
          counts = {}
          paragraph = paragraph.lower()
          banned = set(banned)
          paragraph = re.split(r'[!\?\',;\. ]', paragraph)
          
          for word in paragraph:
              if len(word)==0:
                  continue
  
              if word not in banned:
                  if word in counts:
                      counts[word] += 1
                  else:
                      counts[word] = 1
  
          k = list(counts.keys())
          v = list(counts.values())
  
          idx = v.index(max(v))
          
          return k[idx]
  ```

### 200. 图像渲染

* 栈

  ```python
  class Solution:
      def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
  
  
  
          val = image[sr][sc]
          if val==newColor:
              return image
  
          stack = []
          stack.append((sr, sc))
          while len(stack) > 0:
              r, c = stack.pop()
              if image[r][c]==val:
                  image[r][c] = newColor
  
                  if (c+1) < len(image[0]) and image[r][c+1]==val:
                      stack.append((r, c+1))
                  if r > 0 and image[r-1][c]==val:
                      stack.append((r-1, c))
                  if c > 0 and image[r][c-1]==val:
                      stack.append((r, c-1))
                  if (r+1) < len(image) and image[r+1][c]==val:
                      stack.append((r+1, c))
  
          return image
  ```

* 深度优先搜索

  ```python
  class Solution:
      def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
  
  
  
          val = image[sr][sc]
          if val==newColor:
              return image
  
          def dfs(r, c):
              if image[r][c]==val:
                  image[r][c] = newColor
  
                  if r > 0 and image[r-1][c]==val:
                      dfs(r-1, c)
                  if c > 0 and image[r][c-1]==val:
                      dfs(r, c-1)
                  if r < len(image)-1 and image[r+1][c]==val:
                      dfs(r+1, c)
                  if c < len(image[0])-1 and image[r][c+1]==val:
                      dfs(r, c+1)
          dfs(sr, sc) 
          return image
  ```

### 201. 数据流中的第K大元素

* 排序

  ```python
  class KthLargest:
  
      def __init__(self, k: int, nums: List[int]):
          self.nums = nums
          self.k = k
          self.nums.sort()
  
      def add(self, val: int) -> int:
          self.nums.append(val)
          self.nums.sort()
          return self.nums[-self.k]
  ```

### 202. 设计哈希集合

* 开放链址法

  ```python
  class MyHashSet:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.vals = [None for _ in range(1000)]
  
      def add(self, key: int) -> None:
          if self.contains(key):
              return None
  
          idx = key%1000
          if self.vals[idx]==None:
              self.vals[idx] = [key]
          else:
              self.vals[idx].append(key)
          
  
          
  
      def remove(self, key: int) -> None:
          if self.contains(key):
              idx = key%1000
              self.vals[idx].remove(key)
  
      def contains(self, key: int) -> bool:
          """
          Returns true if this set contains the specified element
          """
          idx = key%1000
          if self.vals[idx] and key in self.vals[idx]:
              return True
          else:
              return False
          
  ```

### 203. 设计哈希映射

* 开放链址

  ```python
  class MyHashMap:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.vals = [None for _ in range(1000)]
          
  
      def put(self, key: int, value: int) -> None:
          """
          value will always be non-negative.
          """
          idx = key%1000
          if self.vals[idx]==None:
              self.vals[idx] = [[key, value]]
          else:
              change = False
              for i in range(len(self.vals[idx])):
                  if self.vals[idx][i][0]==key:
                      self.vals[idx][i][1] = value
                      change = True
                      break
  
              if change==False:
                  self.vals[idx].append([key, value])
              
  
                  
              
  
          
  
          
  
      def get(self, key: int) -> int:
          """
          Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
          """
          idx = key%1000
          if self.vals[idx]==None:
              return -1
          else:
              for i in range(len(self.vals[idx])):
                  if self.vals[idx][i][0]==key:
                      return self.vals[idx][i][1]
  
              return -1
          
  
      def remove(self, key: int) -> None:
          """
          Removes the mapping of the specified value key if this map contains a mapping for the key
          """
          contains = self.get(key)
          idx = key%1000
          if contains!=-1:
              temp = []
              for i in range(len(self.vals[idx])):
                  if self.vals[idx][i][0]==key:
                      self.vals[idx].pop(i)
                      break
  ```

### 204. 1比特与2比特字符

* 暴力

  ```python
  class Solution:
      def isOneBitCharacter(self, bits: List[int]) -> bool:
  
          i = 0
          while i < len(bits):
              if bits[i]==0:
                  if i==len(bits)-1:
                      return True
                  else:
                      i += 1
              else:
                  if i==len(bits)-1:
                      return False
                  else:
                      i += 2
  
          return False
  ```

### 205. 子域名访问计数

* 计数法

  ```python
  class Solution:
      def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
  
          counts = {}
  
          for fdomain in cpdomains:
              freq, domain = fdomain.split(' ')
              i = 0
              while i < len(domain):
                  if domain[i:] in counts:
                      counts[domain[i:]] += int(freq)
                  else:
                      counts[domain[i:]] = int(freq)
                  
                  while i < len(domain) and domain[i]!='.':
                      i += 1
  
                  if i < len(domain):
                      i += 1
  
          ans = []
          for k, v in counts.items():
              ans.append('{} {}'.format(v, k))
  
          return ans
  
              
  
  ```

  



### 206. 矩阵中的幻方

* 暴力

  ```python
  class Solution:
  
  
      def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
  
          if len(grid) < 3 or len(grid[0]) < 3:
              return 0
  
          def isMagic(r, c):
  
              mat = [grid[i][j] for i in range(r, r+3) for j in range(c, c+3)]
              if set(mat)!=set(range(1, 10)):
                  return False
  
              if sum(mat[0:3])==15 and sum(mat[3:6])==15 and sum(mat[6:9])==15 \
              and sum(mat[0:7:3])==15 and sum(mat[1:8:3])==15 and sum(mat[2:9:3])\
              ==15 and sum(mat[0:9:4])==15 and sum(mat[2:7:2])==15:
                  return True
              
              return False
              
  
          ans = 0
          for i in range(len(grid)-2):
              for j in range(len(grid[0])-2):
                  if isMagic(i, j):
                      ans += 1
          
          return ans
  ```

### 207. 比较含退格的字符串

* 栈

  ```python
  class Solution:
      def backspaceCompare(self, S: str, T: str) -> bool:
  
  
          def text(s):
              stack = []
              for i in range(len(s)):
                  if s[i]=='#' and len(stack) > 0:
                      stack.pop()
                  elif s[i]!='#':
                      stack.append(s[i])
              
              return ''.join(stack)
  
          return text(S)==text(T)
  ```

### 208. 到最近人的最大距离

* 暴力

  ```python
  class Solution:
      def maxDistToClosest(self, seats: List[int]) -> int:
  
          dist = 0
  
          i = 0
          while i < len(seats) and seats[i]==0:
              i += 1
          
          dist = max(dist, i)
  
          j = len(seats)-1
          while j >= 0 and seats[j]==0:
              j -= 1
          
          dist = max(dist, len(seats)-1-j)
  
          while i < j:
              
              i += 1
              count = 0
              while i < j and seats[i]==0:
                  count += 1
                  i += 1
  
              dist = max(dist, (count+1)//2)
  
          return dist
  ```

* 列表

  ```python
  class Solution:
      def maxDistToClosest(self, seats: List[int]) -> int:
  
          i = 0
          ans = []
          while i < len(seats):
              while i < len(seats) and seats[i]==1:
                  i += 1
              
              count = 0
              while i < len(seats) and seats[i]==0:
                  i += 1
                  count += 1
  
              if count > 0:
                  ans.append(count)
              
          if seats[0]==0:
              ans[0] *= 2
          if seats[-1]==0:
              ans[-1] *= 2
  
          return (max(ans)+1)//2
  ```

### 209. 山脉数组的峰顶索引

* 扫描

  ```python
  class Solution:
      def peakIndexInMountainArray(self, A: List[int]) -> int:
  
          i = 0
          while i < len(A)-1 and A[i] < A[i+1]:
              i += 1
  
          return i
  ```


### 210. 最小移动次数使数组元素相等

* 公式

  ```python
  class Solution:
      def minMoves(self, nums: List[int]) -> int:
  
          return sum(nums)-min(nums)*len(nums)
  ```

* 动态规划

  ```python
  class Solution:
      def minMoves(self, nums: List[int]) -> int:
          if len(nums) <= 1:
              return 0
          if len(nums)==2:
              return nums[1]-nums[0]
  
          
          nums.sort()
          opt = [0 for _ in range(len(nums))]
          opt[0] = 0
          opt[1] = nums[1]-nums[0]
          for i in range(2, len(nums)):
              nums[i] += opt[i-1]
              opt[i] = opt[i-1] + nums[i]-nums[i-1]
  
          return opt[-1]
  ```

### 211. 把二叉搜索树转换为累加树

* 反序中序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def cumsum(self, root):
          if root==None:
              return
          
          self.cumsum(root.right)
          self.sums += root.val
          root.val = self.sums
          self.cumsum(root.left)
  
  
  
      def convertBST(self, root: TreeNode) -> TreeNode:
  
          self.sums = 0
          self.cumsum(root)
          return root
  ```

### 212. 二叉树的直径

* 计算深度

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def depth(self, root):
          if root==None:
              return -1
          
          return 1 + max(self.depth(root.left), self.depth(root.right))
  
      def diameter(self, root):
          if root==None or root.left==None and root.right==None:
              return 0
          
          if root.left and root.right:
              return max(2+self.depth(root.left)+self.depth(root.right), 
              self.diameter(root.left), self.diameter(root.right))
          elif root.left:
              return max(1+self.depth(root.left), \
              self.diameter(root.left))
          else:
              return max(1+self.depth(root.right), \
              self.diameter(root.right))
          
  
      def diameterOfBinaryTree(self, root: TreeNode) -> int:
  
          return self.diameter(root)
  ```

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def diameterOfBinaryTree(self, root: TreeNode) -> int:
  
          self.ans = 0
          def depth(root):
              if root==None:
                  return 0
              
              L = depth(root.left)
              R = depth(root.right)
  
              self.ans = max(self.ans, L+R)
              return max(L, R) + 1
          depth(root)
          return self.ans
  ```

### 213. N叉树的先序遍历

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def diameterOfBinaryTree(self, root: TreeNode) -> int:
  
          self.ans = 0
          def depth(root):
              if root==None:
                  return 0
              
              L = depth(root.left)
              R = depth(root.right)
  
              self.ans = max(self.ans, L+R)
              return max(L, R) + 1
          depth(root)
          return self.ans
  ```

* 栈

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  class Solution:
      def preorder(self, root: 'Node') -> List[int]:
          if root==None:
              return []
          
          ans = []
          stack = []
          stack.append(root)
          while len(stack) > 0:
              top = stack.pop()
              ans.append(top.val)
              for child in reversed(top.children):
                  if child:
                      stack.append(child)
                  
          return ans
  ```

### 214. N叉树的后序遍历

* 递归

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  class Solution:
      def postorder(self, root: 'Node') -> List[int]:
  
          ans = []
          def post(root):
              if root==None:
                  return
  
              for child in root.children:
                  post(child)
              ans.append(root.val)
  
          post(root)
  
          return ans
  ```

* 栈

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  class Solution:
      def postorder(self, root: 'Node') -> List[int]:
          if root==None:
              return []
  
          stack = []
          ans = []
          stack.append((root, False))
          while len(stack) > 0:
              top, status = stack.pop()
              if top:
                  if status==True:
                      ans.append(top.val)
                  else:
                      stack.append((top, True))
                      for child in reversed(top.children):
                          stack.append((child, False))
  
          return ans
  ```

* 逆序反向先序遍历

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  class Solution:
      def postorder(self, root: 'Node') -> List[int]:
          if root==None:
              return []
  
          stack = []
          ans = []
          stack.append(root)
          while len(stack) > 0:
              top = stack.pop()
              ans.append(top.val)
              for child in top.children:
                  if child:
                      stack.append(child)
  
              
  
          return ans[::-1]
  ```

### 215. N叉树的最大深度

* 递归

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  class Solution:
      def maxDepth(self, root: 'Node') -> int:
  
          if root==None:
              return 0
  
          depth = 0
          for child in root.children:
              depth = max(depth, self.maxDepth(child))
          
          return depth+1
  ```

* 栈

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  class Solution:
      def maxDepth(self, root: 'Node') -> int:
          if root==None:
              return 0
  
          stack = []
          ans = 0
          stack.append((root, 1))
          while len(stack) > 0:
              top, d = stack.pop()
              ans = max(ans, d)
              for child in top.children[::-1]:
                  if child:
                      stack.append((child, d+1))
  
          return ans
  ```

### 216. 学生出勤记录 I

* 暴力

  ```python
  class Solution:
      def checkRecord(self, s: str) -> bool:
          if len(s)<=1:
              return True
  
          absent = 0
          late = 0
          for x in s:
              if x!='L':
                  late = 0
              else:
                  late += 1
  
              if x=='A':
                  absent += 1
  
              if absent > 1 or late > 2:
                  return False
              
  
          return True
  ```

### 217. 合并二叉树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
  
          if t1==None:
              return t2
          if t2==None:
              return t1
  
          t1.val += t2.val
          t1.left = self.mergeTrees(t1.left, t2.left)
          t1.right = self.mergeTrees(t1.right, t2.right)
          return t1
  ```

* 迭代

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
          if t1==None:
              return t2
          if t2==None:
              return t1
          
  
          stack = []
          stack.append((t1, t2))
  
          while len(stack) > 0:
              r1, r2 = stack.pop()
              r1.val += r2.val
              if r1.right and r2.right:
                  stack.append((r1.right, r2.right))
              if r1.left and r2.left:
                  stack.append((r1.left, r2.left))
  
              if r1.left==None:
                  r1.left = r2.left
              if r1.right==None:
                  r1.right = r2.right
  
          return t1
  ```

### 218. 二叉树的坡度

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def findTilt(self, root: TreeNode) -> int:
  
          self.ans = 0
          def sums(root):
              if root==None:
                  return 0
  
              L = sums(root.left)
              R = sums(root.right)
              self.ans += abs(L-R)
              return L+R+root.val
  
          sums(root)
  
          return self.ans
  ```

### 219. 根据二叉树创建字符串

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def tst(self, t, ans):
          if t==None:
              return
  
          ans.append(t.val)
          if t.right==None and t.left==None:
              return
          
          ans.append('(')
          self.tst(t.left, ans)
          ans.append(')')
          if t.right:
              ans.append('(')
              self.tst(t.right, ans)
              ans.append(')')
          
  
              
  
      def tree2str(self, t: TreeNode) -> str:
          ans = []
          self.tst(t, ans)
          return ''.join(map(str,ans))
  ```


### 220. 将有序数组转化为二叉搜索树

* 分治法(递归)

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def balancedTree(self, nums, left, right):
          if left > right:
              return None
  
          if left==right:
              root = TreeNode(nums[left])
              return root
          
          mid = (left + right + 1)//2
          root = TreeNode(nums[mid])
          root.left = self.balancedTree(nums, left, mid-1)
          root.right = self.balancedTree(nums, mid+1, right)
          return root
  
  
      def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
  
          return self.balancedTree(nums, 0, len(nums)-1)
  ```

### 221. 路径总和III

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      
  
      def dfs(self, root, sums, target):
          if root==None:
              return
          
          sums += root.val
          if sums==target:
              self.ans += 1
          self.dfs(root.left, sums, target)
          self.dfs(root.right, sums, target)
  
      def prefix(self, root, target):
          if root==None:
              return
          
          self.dfs(root, 0, target)
          self.prefix(root.left, target)
          self.prefix(root.right, target)
  
  
  
      def pathSum(self, root: TreeNode, sum: int) -> int:
          
          self.ans = 0
          self.prefix(root, sum)
          return self.ans
  ```

### 222. 另一个树的子树

* 暴力

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def isSame(self, s, t):
          if s==None and t==None:
              return True
  
          if s==None or t==None:
              return False
  
          if s.val==t.val:
              return self.isSame(s.left, t.left) and self.isSame(s.right, t.right)
  
      def dfs(self, s, t):
          if self.isSame(s, t):
              return True
          
          if s==None:
              return False
          
          return self.dfs(s.left, t) or self.dfs(s.right, t)
  
      def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
          
          return self.dfs(s, t)
  ```

* 先序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, ans):
          if root==None:
              ans.append('#')
              return
          
          ans.append(root.val)
          self.dfs(root.left, ans)
          self.dfs(root.right, ans)
  
      def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
          
          st = []
          tt = []
          self.dfs(s, st)
          self.dfs(t, tt)
  
          i = 0
          while i < len(st):
              
              while i < len(st) and st[i]!=tt[0]:
                  i += 1
  
              temp = i
              if i >= len(st):
                  return False
  
              j = 0
              while i < len(st) and j < len(tt) and st[i]==tt[j]:
                  i += 1
                  j += 1
  
              if j >= len(tt):
                  return True
              
              i = temp + 1
  
          return False
  ```

### 223. 二叉树的层平均值

* 队列

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import queue
  class Solution:
      def averageOfLevels(self, root: TreeNode) -> List[float]:
          if root==None:
              return []
  
          q = queue.Queue()
          q.put(root)
          layer = 1
          ans = []
          while not q.empty():
              
              sums = 0
              count = 0
              for _ in range(layer):
                  node = q.get()
                  sums += node.val
                  if node.left:
                      q.put(node.left)
                      count += 1
                  if node.right:
                      q.put(node.right)
                      count += 1
  
              ans.append(sums/layer)
              layer = count
  
          return ans
  ```

* 深度优先搜索

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import queue
  class Solution:
  
      def dfs(self, root, h, sums, counts):
          if root==None:
              return
  
          if len(sums) < h:
              sums.append(root.val)
              counts.append(1)
          else:
              sums[h-1] += root.val
              counts[h-1] += 1
  
          self.dfs(root.left, h+1, sums, counts)
          self.dfs(root.right, h+1, sums, counts)
  
  
      def averageOfLevels(self, root: TreeNode) -> List[float]:
          
          sums = []
          counts = []
          self.dfs(root, 1, sums, counts)
  
          for i in range(len(sums)):
              sums[i] /= counts[i]
  
          return sums
  ```

### 224. 两数之和IV

* 中序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
  
      def dfs(self, root, k, counts):
          if root==None:
              return
  
          self.dfs(root.left, k, counts)
  
          if k-root.val in counts:
              self.ans = True
              return
          counts.add(root.val)
  
          self.dfs(root.right, k, counts)
  
      def findTarget(self, root: TreeNode, k: int) -> bool:
  
          counts = set()
          self.ans = False
          self.dfs(root, k, counts)
          return self.ans
  ```



### 224. 修剪二叉搜索树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def trim(self, root, L, R):
          if root==None:
              return None
          
          if root.val < L:
              return self.trim(root.right, L, R)
          if root.val > R:
              return self.trim(root.left, L, R)
          
          root.left = self.trim(root.left, L, R)
          root.right = self.trim(root.right, L, R)
  
          return root
          
  
      def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
  
          return self.trim(root, L, R)
  ```

### 225. 二叉树中第二小的节点

* 暴力

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, vals):
          if root==None:
              return
  
          vals.add(root.val)
          self.dfs(root.left, vals)
          self.dfs(root.right, vals)
  
      def findSecondMinimumValue(self, root: TreeNode) -> int:
          
          vals = set()
          self.dfs(root, vals)
          vals.remove(min(vals))
          return min(vals) if len(vals) > 0 else -1 
  ```

* 深度优先搜索

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return
          
          if root.val!=self.min_v and (self.sec_v==self.min_v or self.sec_v > root.val):
              self.sec_v = root.val
  
          self.dfs(root.left)
          self.dfs(root.right)
  
  
      def findSecondMinimumValue(self, root: TreeNode) -> int:
  
          self.min_v = root.val
          self.sec_v = root.val
          self.dfs(root)
          return -1 if self.sec_v==root.val else self.sec_v
  ```

### 226. 链表的中间节点

* 快慢指针

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def middleNode(self, head: ListNode) -> ListNode:
          if head==None:
              return None
  
          p = head
          q = head
  
          while p.next:
              p = p.next
              if p.next:
                  p = p.next
              q = q.next
  
          return q
  ```

### 227. 三维形体投影面积

* 暴力

  ```python
  class Solution:
      def projectionArea(self, grid: List[List[int]]) -> int:
  
          xy = 0
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  xy += 1 if grid[i][j] > 0 else 0
  
          yz = 0
          for j in range(len(grid[0])):
              col = 0
              for i in range(len(grid)):
                  col = max(col, grid[i][j])
              yz += col
  
          xz = 0
          for i in range(len(grid)):
              xz += max(grid[i])
  
          return xy + yz + xz
  ```

### 228. 统计有序矩阵中的负数

* 右上角

  ```python
  class Solution:
      def countNegatives(self, grid: List[List[int]]) -> int:
  
          i = 0
          j = len(grid[0])-1
          
          count = 0
          while i < len(grid) and j >= 0:
              if grid[i][j] >= 0:
                  i += 1
              else:
                  count += len(grid)-i
                  j -= 1
  
          return count
  ```

### 229. 判断字符是否唯一

* 暴力

  ```python
  class Solution:
      def isUnique(self, astr: str) -> bool:
          if len(astr)==1:
              return True
  
          for i in range(len(astr)-1):
              for j in range(i+1, len(astr)):
                  if astr[i]==astr[j]:
                      return False
  
          return True
  ```

* 集合

  ```python
  class Solution:
      def isUnique(self, astr: str) -> bool:
          chars = set()
          for s in astr:
              if s in chars:
                  return False
              else:
                  chars.add(s)
  
          return True
  ```


### 230. 判定是否互为字符重排

* 排序

  ```python
  class Solution:
      def CheckPermutation(self, s1: str, s2: str) -> bool:
  
          s1 = list(s1)
          s2 = list(s2)
  
          s1.sort()
          s1 = ''.join(s1)
  
          s2.sort()
          s2 = ''.join(s2)
  
          return s1==s2
  ```

* 计数法

  ```python
  class Solution:
      def CheckPermutation(self, s1: str, s2: str) -> bool:
  
          c1 = {}
          for s in s1:
              if s in c1:
                  c1[s] += 1
              else:
                  c1[s] = 1
  
          for s in s2:
              if s not in c1:
                  return False
              else:
                  c1[s] -= 1
          
          for k, v in c1.items():
              if v > 0 or v < 0:
                  return False
  
          return True
  ```

### 231. URL化

* 暴力

  ```python
  class Solution:
      def CheckPermutation(self, s1: str, s2: str) -> bool:
  
          c1 = {}
          for s in s1:
              if s in c1:
                  c1[s] += 1
              else:
                  c1[s] = 1
  
          for s in s2:
              if s not in c1:
                  return False
              else:
                  c1[s] -= 1
          
          for k, v in c1.items():
              if v > 0 or v < 0:
                  return False
  
          return True
  ```

### 232. 回文排列

* 计数法

  ```python
  class Solution:
  
  
      def canPermutePalindrome(self, s: str) -> bool:
          counts = {}
          for x in s:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          ans = 0
          for v in counts.values():
              if v%2==1:
                  ans += 1
  
          return ans <= 1
  ```

* 集合

  ```python
  class Solution:
  
  
      def canPermutePalindrome(self, s: str) -> bool:
          counts = set()
          for x in s:
              if x in counts:
                  counts.remove(x)
              else:
                  counts.add(x)
  
          return len(counts) <= 1
  
  ```

### 233. 字符串压缩

* 暴力

  ```python
  class Solution:
  
  
      def canPermutePalindrome(self, s: str) -> bool:
          counts = set()
          for x in s:
              if x in counts:
                  counts.remove(x)
              else:
                  counts.add(x)
  
          return len(counts) <= 1
  
  ```

### 234. 字符串轮转

* 暴力

  ```python
  class Solution:
      def isFlipedString(self, s1: str, s2: str) -> bool:
          if len(s1)!=len(s2):
              return False
          if len(s1)==0 and len(s2)==0:
              return True
  
          i = 0
          while i < len(s2):
              
              while i < len(s2) and s2[i]!=s1[0]:
                  i += 1
  
              if i >= len(s2):
                  return False
              
              if s2[i:]==s1[:len(s2)-i] and s2[:i]==s1[len(s2)-i:]:
                  return True
  
              i += 1
  
          return False
  ```

* 轮转性质

  ```python
  class Solution:
      def isFlipedString(self, s1: str, s2: str) -> bool:
          if len(s1)!=len(s2):
              return False
              
          s2 += s2
          return s1 in s2
  ```

### 235. 移除重复节点

* 集合

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def removeDuplicateNodes(self, head: ListNode) -> ListNode:
          if head==None:
              return None
  
          nodes = set()
          p = head
          nodes.add(p.val)
          while p.next:
              if p.next.val in nodes:
                  p.next = p.next.next
              else:
                  nodes.add(p.next.val)
                  p = p.next
  
          return head
  ```

### 236. 删除中间节点

* 暴力

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def deleteNode(self, node):
          """
          :type node: ListNode
          :rtype: void Do not return anything, modify node in-place instead.
          """
          node.val = node.next.val
          node.next = node.next.next
  ```

### 237. 回文链表

* 暴力

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
  
      def ispalind(self, ans):
          if len(ans)==0:
              return True
          i = 0
          j = len(ans)-1
          while i < j and ans[i]==ans[j]:
              i += 1
              j -= 1
          
          return i >= j
  
      def isPalindrome(self, head: ListNode) -> bool:
  
          p = head
          ans = []
          while p:
              ans.append(p.val)
              p = p.next
  
          return self.ispalind(ans)
  ```

### 238. 翻转数位

* 暴力

  ```python
  class Solution:
      def reverseBits(self, num: int) -> int:
          
          max_v = 0
          prev = 0
          while num:
  
              c0 = 0
              while num and num%2==0:
                  c0 += 1
                  num //= 2
  
              c1 = 0
              while num and num%2==1:
                  c1 += 1
                  num //= 2
  
              if c0==1:
                  max_v = max(prev+c1, max_v)
              else:
                  max_v = max(c1, max_v)
              prev = c1
  
          return max_v+1
  ```

### 239. 合并排序的数组

* 倒序

  ```python
  class Solution:
      def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
          """
          Do not return anything, modify A in-place instead.
          """
  
          k = m + n - 1
          i = m - 1
          j = n - 1
          while i >= 0 and j >= 0:
              if A[i] > B[j]:
                  A[k] = A[i]
                  i -= 1
              else:
                  A[k] = B[j]
                  j -= 1
              k -= 1
  
          while i >= 0:
              A[k] = A[i]
              i -= 1
              k -= 1
          while j >= 0:
              A[k] = B[j]
              j -= 1
              k -= 1
  ```


### 240. 栈的最小值

* 暴力

  ```python
  class MinStack:
  
      def __init__(self):
          """
          initialize your data structure here.
          """
          self.stack = []
          self.min_v = None
          
  
      def push(self, x: int) -> None:
          self.stack.append(x)
          self.min_v = min(self.stack)
          
  
      def pop(self) -> None:
          self.stack.pop()
          self.min_v = min(self.stack) if len(self.stack) > 0 else None
          
  
      def top(self) -> int:
          if len(self.stack) > 0:
              return self.stack[-1]
          
  
      def getMin(self) -> int:
          return self.min_v
          
  
  
  # Your MinStack object will be instantiated and called as such:
  # obj = MinStack()
  # obj.push(x)
  # obj.pop()
  # param_3 = obj.top()
  # param_4 = obj.getMin()
  ```

### 241. 阶乘尾数

* 5的个数

  ```python
  class Solution:
      def trailingZeroes(self, n: int) -> int:
  
          sums = 0
          k = 5
          while k <= n:
              sums += n//k
              k *= 5
  
          return sums
  ```

### 242. 化栈为队

* 双栈法

  ```python
  class MyQueue:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.stk1 = []
          self.stk2 = []
  
  
      def push(self, x: int) -> None:
          """
          Push element x to the back of queue.
          """
          while len(self.stk1) > 0:
              self.stk2.append(self.stk1.pop())
  
          self.stk1.append(x)
  
          while len(self.stk2) > 0:
              self.stk1.append(self.stk2.pop())
  
      def pop(self) -> int:
          """
          Removes the element from in front of queue and returns that element.
          """
          return self.stk1.pop()
  
  
      def peek(self) -> int:
          """
          Get the front element.
          """
          return self.stk1[-1]
  
  
      def empty(self) -> bool:
          """
          Returns whether the queue is empty.
          """
          return len(self.stk1)==0
  
  
  
  # Your MyQueue object will be instantiated and called as such:
  # obj = MyQueue()
  # obj.push(x)
  # param_2 = obj.pop()
  # param_3 = obj.peek()
  # param_4 = obj.empty()
  ```

### 243. 最小高度树

* 分治法

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def bst(self, nums, left, right):
          if left==right:
              return TreeNode(nums[left])
  
          mid = (left + right)//2
          root = TreeNode(nums[mid])
          if left < mid:
              root.left = self.bst(nums, left, mid-1)
          if mid < right:
              root.right = self.bst(nums, mid+1, right)
          
          return root
  
      def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
          if len(nums)==0:
              return None
  
          return self.bst(nums, 0, len(nums)-1)
  ```

### 244. 检查平衡性

* 计算高度

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def depth(self, root):
          if root==None:
              return 0
          
          return max(self.depth(root.left), self.depth(root.right)) + 1
  
      def isBalanced(self, root: TreeNode) -> bool:
          if root==None:
              return True
  
          if abs(self.depth(root.left)-self.depth(root.right)) > 1:
              return False
  
          return self.isBalanced(root.left) and self.isBalanced(root.right)
  ```

* 计算高度2

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def depth(self, root, d):
          if root==None:
              return d
          
          return max(self.depth(root.left, d+1), self.depth(root.right, d+1))
  
      def isBalanced(self, root: TreeNode) -> bool:
          if root==None:
              return True
  
          if abs(self.depth(root.left, 0)-self.depth(root.right, 0)) > 1:
              return False
  
          return self.isBalanced(root.left) and self.isBalanced(root.right)
  ```

### 245. 三步问题

* 动态规划

  ```python
  class Solution:
      def waysToStep(self, n: int) -> int:
  
          if n==1:
              return 1
          if n==2:
              return 2
          if n==3:
              return 4
  
          f1 = 1
          f2 = 2
          f3 = 4
          for _ in range(3, n):
              temp1, temp2 = f2, f3
              f3 += (f1 + f2)%1000000007
              f1, f2 = temp1, temp2
  
          return f3%1000000007
  ```

### 246. 魔术索引

* 暴力搜索

  ```python
  class Solution:
      def findMagicIndex(self, nums: List[int]) -> int:
  
          for i in range(len(nums)):
              if nums[i]==i:
                  return i
  
          return -1
  ```

### 247. 汉诺塔问题

* 分治法

  ```python
  class Solution:
  
      def move(self, n, A, B, C):
          if n==1:
              C.append(A.pop())
          else:
              self.move(n-1, A, C, B)
              C.append(A.pop())
              self.move(n-1, B, A, C)
  
  
      def hanota(self, A: List[int], B: List[int], C: List[int]) -> None:
          """
          Do not return anything, modify C in-place instead.
          """
  
          self.move(len(A), A, B, C)
  ```

### 248. 颜色填充

* 栈

  ```python
  class Solution:
      def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
  
  
          stack = []
          stack.append((sr, sc))
          val = image[sr][sc]
  
          while len(stack) > 0:
              i, j = stack.pop()
              if image[i][j]!=newColor:
                  
                  image[i][j] = newColor
                  
                  if i > 0 and image[i-1][j]==val:
                      stack.append((i-1, j))
                  if i < len(image)-1 and image[i+1][j]==val:
                      stack.append((i+1, j))
                  if j > 0 and image[i][j-1]==val:
                      stack.append((i, j-1))
                  if j < len(image[0])-1 and image[i][j+1]==val:
                      stack.append((i, j+1))
  
          return image
  ```

### 249. 稀疏数组搜索

* 线性扫描

  ```python
  class Solution:
      def findString(self, words: List[str], s: str) -> int:
  
          for i, w in enumerate(words):
              if len(w)==0:
                  continue
  
              if w==s:
                  return i
  
          return -1
  ```

* 二分法

  ```python
  class Solution:
      def findString(self, words: List[str], s: str) -> int:
  
          left = 0
          right = len(words)-1
          while left <= right:
              
              while left <= right and words[left]=='':
                  left += 1
              while left <= right and words[right]=='':
                  right -= 1
  
              if left <= right:
                  mid = (left + right)//2
                  while mid >= left and words[mid]=='':
                      mid -= 1
  
                  if words[mid]==s:
                      return mid
                  elif words[mid] < s:
                      left = mid + 1
                  else:
                      right = mid - 1
  
          return -1
  ```

### 250. 最大数值

* 位运算表示绝对值

  ```python
  class Solution:
      def maximum(self, a: int, b: int) -> int:
  
          sub = a-b
          sums = 0
          sums += (sub ^ sub >> 63) - (sub >> 63)
          sums += a + b
  
          return sums//2
  ```

### 251. 跳水版

* 规律

  ```python
  class Solution:
      def divingBoard(self, shorter: int, longer: int, k: int) -> List[int]:
          if k==0:
              return []
          
          if longer==shorter:
              return [longer*k]
  
          dist = longer - shorter
          ans = [v for v in range(k*shorter, k*longer+1, dist)]
  
          return ans
  ```

### 252. 连续数列

* 动态规划

  ```python
  class Solution:
      def maxSubArray(self, nums: List[int]) -> int:
          if len(nums)==0:
              return None
  
          sums = 0
          ans = nums[0]
          for num in nums:
              if sums > 0:
                  sums += num
              else:
                  sums = num
              
              ans = max(ans, sums)
  ```

* 原始动态规划

  ```python
  class Solution:
      def maxSubArray(self, nums: List[int]) -> int:
          if len(nums)==0:
              return None
  
          opt = [0 for _ in range(len(nums))]
          opt[0] = nums[0]
  
          for i in range(1, len(nums)):
              if opt[i-1] > 0:
                  opt[i] = opt[i-1] + nums[i]
              else:
                  opt[i] = nums[i]
  
          return max(opt)
  ```

* 动态规划优化

  ```python
  class Solution:
      def maxSubArray(self, nums: List[int]) -> int:
          if len(nums)==0:
              return None
  
          f0 = nums[0]
          ans = f0
          for i in range(1, len(nums)):
              if f0 > 0:
                  f0 += nums[i]
              else:
                  f0 = nums[i]
  
              ans = max(f0, ans)
  
          return ans
  ```

* 分治法

  ```python
  class Solution:
  
      def maxsub(self, nums, left, right):
          if left==right:
              return nums[left]
          
          mid = (left + right)//2
          lv = self.maxsub(nums, left, mid)
          rv = self.maxsub(nums, mid+1, right)
  
          sums = 0
          i = mid
          maxv = nums[i]
          while i >= left:
              sums += nums[i]
              i -= 1
              maxv = max(maxv, sums)
          
          lm = maxv
  
          sums = 0
          i = mid+1
          maxv = nums[i]
          while i <= right:
              sums += nums[i]
              i += 1
              maxv = max(maxv, sums)
  
          rm = maxv
  
          return max([lv, rv, lm+rm])
  
      def maxSubArray(self, nums: List[int]) -> int:
  
          return self.maxsub(nums, 0, len(nums)-1)
  ```

### 253. 消失的数字

* 公式法

  ```python
  class Solution:
      def missingNumber(self, nums: List[int]) -> int:
  
          n = len(nums)
          return n*(n+1)//2 - sum(nums)
  ```

* 桶排序

  ```python
  class Solution:
      def missingNumber(self, nums: List[int]) -> int:
  
          i = 0
          while i < len(nums):
              if i==nums[i] or nums[i]==len(nums):
                  i += 1
              else:
                  idx = nums[i]
                  temp = nums[i]
                  nums[i] = nums[idx]
                  nums[idx] = temp
  
          i = 0
          while i < len(nums):
              if nums[i]!=i:
                  return i
              i += 1
  
          return len(nums)
  ```

* 异或

  ```python
  class Solution:
      def missingNumber(self, nums: List[int]) -> int:
  
          ans = 0
          for i in range(len(nums)+1):
              ans ^= i
          
          for i in range(len(nums)):
              ans ^= nums[i]
  
          return ans
  ```

### 254. 主要元素

* 计数法

  ```python
  class Solution:
      def majorityElement(self, nums: List[int]) -> int:
  
          counts = {}
          for num in nums:
              if num in counts:
                  counts[num] += 1
              else:
                  counts[num] = 1
  
          for k, v in counts.items():
              if 2*v > len(nums):
                  return k
  
          return -1
  ```

* 排序法

  ```python
  class Solution:
      def majorityElement(self, nums: List[int]) -> int:
  
          nums.sort()
          return nums[len(nums)//2]
  ```

* 摩尔投票法

  ```python
  class Solution:
      def majorityElement(self, nums: List[int]) -> int:
  
          val = 0
          count = 0
          for num in nums:
              if count==0:
                  val = num
                  count += 1
              elif val==num:
                  count += 1
              else:
                  count -= 1
  
          return val
  ```

### 255. BiNode

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return
  
          self.dfs(root.left)
          root.left = None
          self.prev.right = root
          self.prev = root 
          self.dfs(root.right)
  
      def convertBiNode(self, root: TreeNode) -> TreeNode:
  
          newroot = TreeNode(0)
          self.prev = newroot
          self.dfs(root)
          return newroot.right
  ```

### 256. 从上到下打印二叉树

* 队列

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import queue
  class Solution:
  
      def levelOrder(self, root: TreeNode) -> List[List[int]]:
          if root==None:
              return []
  
          q = queue.Queue()
          q.put((root, 1))
          ans = []
  
          while not q.empty():
              node, d = q.get()
              if d > len(ans):
                  ans.append([node.val])
              else:
                  ans[d-1].append(node.val)
              
              if node.left:
                  q.put((node.left, d+1))
              if node.right:
                  q.put((node.right, d+1))
  
          return ans
  ```

* 先序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  import queue
  class Solution:
  
      def dfs(self, root, d, ans):
          if root==None:
              return
  
          if d > len(ans):
              ans.append([root.val])
          else:
              ans[d-1].append(root.val)
  
          self.dfs(root.left, d+1, ans)
          self.dfs(root.right, d+1, ans)
  
      def levelOrder(self, root: TreeNode) -> List[List[int]]:
  
          ans = []
          self.dfs(root, 1, ans)
          return ans
  ```

### 257. 二叉搜索树中第k大节点

* 中序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, vals, k):
          if root==None:
              return
          
          self.dfs(root.right, vals, k)
          vals.append(root.val)
          self.dfs(root.left, vals, k)
  
      def kthLargest(self, root: TreeNode, k: int) -> int:
          if root==None:
              return None
          vals = []
          self.dfs(root, vals, k)
          return vals[k-1]
  ```

### 258. 构建累乘数组

* 左右累乘

  ```python
  class Solution:
      def constructArr(self, a: List[int]) -> List[int]:
          if len(a)==0:
              return []
  
          left = a[:]
          left[0] = 1
          for i in range(1, len(left)):
              left[i] = left[i-1]*a[i-1]
  
          right = a[:]
          right[-1] = 1
          for i in range(len(right)-2, -1, -1):
              right[i] = right[i+1]*a[i+1]
  
          ans = [left[i]*right[i] for i in range(len(a))]
  
          print(left, right)
          return ans
  ```

### 259. 两个栈实现队列

* 暴力

  ```python
  class CQueue:
  
      def __init__(self):
          self.stk1 = []
  
      def appendTail(self, value: int) -> None:
          if len(self.stk1)==0:
              self.stk1.append(value)
          else:
              temp = []
              while len(self.stk1) > 0:
                  temp.append(self.stk1.pop())
              self.stk1.append(value)
              while len(temp) > 0:
                  self.stk1.append(temp.pop())
  
  
      def deleteHead(self) -> int:
          if len(self.stk1)==0:
              return -1
          
          return self.stk1.pop()
  ```

### 260. 圆圈中最后剩下的数字

* 暴力

  ```python
  class Solution:
      def lastRemaining(self, n: int, m: int) -> int:
  
          nums = list(range(n))
          i = 0
          while len(nums) > 1:
              i = (i+m-1)%len(nums)
              nums.pop(i)
  
          return nums[0]
  ```


### 261. 根据数字二进制下 1 的数目排序

* 自定义排序

  ```python
  from functools import cmp_to_key
  class Solution:
  
      def count(self, x):
          c = 0
          while x:
              c += x%2
              x //= 2
  
          return c
  
      def compare(self, a, b):
          if a==b:
              return 0
  
          if self.count(a) > self.count(b):
              return 1
          elif self.count(a)==self.count(b):
              return 1 if a > b else -1
          else:
              return -1
  
      def sortByBits(self, arr: List[int]) -> List[int]:
  
          arr = sorted(arr, key=cmp_to_key(self.compare))
          return arr
  ```

### 262. 二叉搜索树的最近公共祖先

* 非递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
          if root==None:
              return None
  
          h = root
  
          while h:
              if h.val > p.val and h.val < q.val or h.val > q.val and h.val < p.val:
                  return h
              elif h.val == p.val:
                  return p
              elif h.val == q.val:
                  return q
              elif p.val > h.val and q.val > h.val:
                  h = h.right
              else:
                  h = h.left
  
          return None
  ```

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def ancester(self, root, p, q):
          if p.val > root.val and q.val < root.val or q.val > root.val and p.val < root.val:
              return root
          
          if p.val==root.val or q.val==root.val:
              return root
          
          if p.val > root.val and q.val > root.val:
              return self.ancester(root.right, p, q)
  
          if p.val < root.val and q.val < root.val:
              return self.ancester(root.left, p, q)
  
      def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
          if root==None:
              return None
          
          return self.ancester(root, p, q)
  ```

  

### 263. 二叉树的最近公共祖先

* 记录路径

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def ancester(self, root, p, path, ans):
          if root==None:
              return
          
          path.append(root)
          if p.val==root.val:
              ans.append(path[:])
          else:
              self.ancester(root.left, p, path[:], ans)
              self.ancester(root.right, p, path[:], ans)
  
      def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
  
          if root==None:
              return None
  
          path1 = []
          self.ancester(root, p, [], path1)
          path2 = []
          self.ancester(root, q, [], path2)
  
          if len(path1)==0 or len(path2)==0:
              return None
          
          path1 = set(path1[0])
          path2 = path2[0]
  
          for node in path2[::-1]:
              if node in path1:
                  return node
  ```

### 264. 数组中出现次数超过一半的数字

* 摩尔投票法

  ```python
  class Solution:
      def majorityElement(self, nums: List[int]) -> int:
  
          count = 0
          prev = None
          for num in nums:
              if count==0:
                  prev = num
                  count += 1
              elif prev==num:
                  count += 1
              else:
                  count -= 1
  
          return prev
  
  ```

### 265. 按摩师

* 动态规划

  ```python
  class Solution:
      def massage(self, nums: List[int]) -> int:
          if len(nums)==0:
              return 0
          if len(nums)==1:
              return nums[0]
          if len(nums)==2:
              return max(nums)
          
          opt = [0 for _ in range(len(nums))]
          opt[0] = nums[0]
          opt[1] = max(nums[0], nums[1])
  
          for i in range(2, len(nums)):
              opt[i] = max(nums[i] + opt[i-2], opt[i-1])
  
          return opt[-1]
  ```

### 266. 珠玑妙算

* 计数法

  ```python
  class Solution:
      def masterMind(self, solution: str, guess: str) -> List[int]:
  
          c1 = {}
          c2 = {}
          ans0, ans1 = 0, 0
          for i in range(len(solution)):
              if solution[i]==guess[i]:
                  ans0 += 1
              else:
                  c1[solution[i]] = 1 if solution[i] not in c1 else c1[solution[i]]+1
                  c2[guess[i]] = 1 if guess[i] not in c2 else c2[guess[i]]+1
  
          for k, v in c1.items():
              if k in c2:
                  ans1 += min(v, c2[k])
  
          return [ans0, ans1]
  ```

### 267. n个骰子的点数

* 动态规划

  ```python
  class Solution:
      def twoSum(self, n: int) -> List[float]:
  
          prob = [[0 for _ in range(6*n+1)] for _ in range(n+1)]
  
          for i in range(1, 7):
              prob[1][i] = 1
  
          for i in range(2, n+1):
              for j in range(i, 6*i+1):
                  k = j-1
                  sums = 0
                  while k >= 0 and k >= j-6:
                      sums += prob[i-1][k]
                      k -= 1
  
                  prob[i][j] = sums
  
          count = sum(prob[n][n:6*n+1])
          ans = [v/count for v in prob[n][n:6*n+1]]
          return ans
  ```

### 268. 整数转换

* python特殊处理

  ```python
  class Solution:
      def convertInteger(self, A: int, B: int) -> int:
  
          if A == B:
              return 0
          if A < 0:
              A &= 0xffffffff
          if B < 0:
              B &= 0xffffffff
          return str(bin(A ^ B)).count("1")
  ```

### 269. 最小栈

* 同步辅助栈

  ```python
  class MinStack:
  
      def __init__(self):
          """
          initialize your data structure here.
          """
          self.data = []
          self.stk = []
  
      def push(self, x: int) -> None:
          self.data.append(x)
          if len(self.stk)==0 or x <= self.stk[-1]:
              self.stk.append(x)
          else:
              self.stk.append(self.stk[-1])
          
  
      def pop(self) -> None:
          if len(self.data) > 0:
              self.stk.pop()
              return self.data.pop()
          
  
      def top(self) -> int:
          return self.data[-1]
          
  
      def getMin(self) -> int:
          return self.stk[-1]
  ```

* 辅助非同步栈

  ```python
  class MinStack:
  
      def __init__(self):
          """
          initialize your data structure here.
          """
          self.data = []
          self.stk = []
  
      def push(self, x: int) -> None:
          self.data.append(x)
          if len(self.stk)==0 or x <= self.stk[-1]:
              self.stk.append(x)
          
      def pop(self) -> None:
          if len(self.data) > 0:
              top = self.data.pop()
              if top==self.stk[-1]:
                  self.stk.pop()
  
              return top
          
  
      def top(self) -> int:
          return self.data[-1]
          
  
      def getMin(self) -> int:
          return self.stk[-1]
          
  
  ```

  

### 270. 最长同值路径

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root):
          if root==None:
              return 0
  
          lm = self.depth(root.left, root.val)
          rm = self.depth(root.right, root.val)
          length = lm + rm
  
  
          return max(length, self.dfs(root.left), self.dfs(root.right))
  
  
      def depth(self, root, v):
          if root==None or root.val!=v:
              return 0
          
          return max(self.depth(root.left, v), self.depth(root.right, v)) + 1
  
      def longestUnivaluePath(self, root: TreeNode) -> int:
  
          return self.dfs(root)
  ```

### 271. 二叉树的堂兄弟节点

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
  
      def fnode(self, root, p):
          if root==None or root.left==None and root.right==None:
              return None
          
          if root.left and root.left.val==p or root.right and root.right.val==p:
              return root
          
          left = self.fnode(root.left, p)
          right = self.fnode(root.right, p)
          
          return left if left else right
          
      def depth(self, root, x, d):
          if root==None:
              return -1
  
          if root.val==x:
              return d
          
          left = self.depth(root.left, x, d+1)
          right = self.depth(root.right, x, d+1)
  
          return max(left, right)
  
  
      def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
  
          fx = self.fnode(root, x)
          dx = self.depth(root, x, 0)
          fy = self.fnode(root, y)
          dy = self.depth(root, y, 0)
  
          return dx==dy and fy!=fx
  ```


### 272. 最近的请求次数

* 双向队列

  ```python
  from collections import deque
  class RecentCounter:
  
      def __init__(self):
          self.q = deque()
  
      def ping(self, t: int) -> int:
          self.q.append(t)
  
          while len(self.q) > 0:
              font = self.q.popleft()
              if font >= t-3000:
                  self.q.appendleft(font)
                  break
          return len(self.q)
  ```

  

### 273. 两个数组间的距离值

* 暴力

  ```python
  class Solution:
      def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
  
          count = 0
          for x in arr1:
              valid = True
              for y in arr2:
                  if y >= x-d and y <= x+d:
                      valid = False
                      break
  
              if valid:
                  count += 1
  
          return count
  ```

* 二分查找

  ```python
  class Solution:
  
      def isValid(self, x, arr2, d):
  
          left = 0
          right = len(arr2)-1
          while left < right:
              mid = (left + right)//2
              if arr2[mid] >= x-d and arr2[mid] <= x+d:
                  return False
              elif arr2[mid] < x-d:
                  left = mid + 1
              else:
                  right = mid - 1
  
          if arr2[right] >= x-d and arr2[right] <= x+d:
              return False
          else:
              return True
  
      def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
  
          arr2.sort()
          count = 0
          for x in arr1:
              if self.isValid(x, arr2, d):
                  
                  count += 1
          return count
  ```

### 274. 找出数组中的幸运数

* 计数法

  ```python
  class Solution:
      def findLucky(self, arr: List[int]) -> int:
  
          counts = {}
          for x in arr:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          max_v = -1
          for k, v in counts.items():
              if k==v:
                  max_v = max(max_v, v)
          return max_v
  ```

### 275. 统计最大组的数目

* 计数法

  ```python
  class Solution:
      
      def digitSum(self, x):
          ans = 0
          while x > 0:
              ans += x%10
              x //= 10
  
          return ans
      
      def countLargestGroup(self, n: int) -> int:
  
          counts = {}
          for x in range(1, n+1):
              val = self.digitSum(x)
              if val in counts:
                  counts[val] += 1
              else:
                  counts[val] = 1
  
          ans = 0
          max_size = max(counts.values())
          for v in counts.values():
              if v==max_size:
                  ans += 1
  
          return ans
  ```

  

### 276. 非递增顺序的最小子序列

* 排序

  ```python
  class Solution:
      def minSubsequence(self, nums: List[int]) -> List[int]:
  
          nums.sort(reverse=True)
  
          sums = sum(nums)
  
          ans = []
          s = 0
          for num in nums:
              s += num
              ans.append(num)
  
              if s*2 > sums:
                  break
          
          return ans
  ```

  

### 277. 按既定顺序创建目标数组

* 插入

  ```python
  class Solution:
      def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
  
          ans = [-1 for _ in range(len(nums))]
  
          for i in range(len(index)):
              j = index[i]
              if ans[j]==-1:
                  ans[j] = nums[i]
              else:
                  for k in range(len(nums)-1, j, -1):
                      ans[k] = ans[k-1]
                  ans[j] = nums[i]
  
          return ans
  ```



### 278. 逐步求和得到正数的最小值

* 暴力

  ```python
  class Solution:
      def minStartValue(self, nums: List[int]) -> int:
  
  
          lowest = nums[0]
          sums = 0
          for num in nums:
              sums += num
              lowest = min(lowest, sums)
  
  
          if lowest >= 1:
              return 1
          else:
              return 1 - lowest
  
  ```

### 279. 拥有最多糖果的孩子

* 暴力

  ```python
  
  class Solution:
      def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
  
          most = max(candies)
  
          ans = []
  
          for candy in candies:
              if candy + extraCandies >= most:
                  ans.append(True)
              else:
                  ans.append(False)
  
          return ans
  ```



### 280. 数组中两个元素的最大乘积

* 堆排序

  ```python
  class Solution:
  
      def swap(self, nums, i, j):
          temp = nums[i]
          nums[i] = nums[j]
          nums[j] = temp
  
      def build_heap(self, heap, n):
          i = n//2 - 1
          while i >= 0:
              self.heapify(heap, i, n)
              i -= 1
  
      def heapify(self, heap, i, n):
          left = 2 * i + 1
          right = 2 * i + 2
          maxi = i
  
          if left < n and heap[maxi] < heap[left]:
              maxi = left
          if right < n and heap[maxi] < heap[right]:
              maxi = right
  
          if maxi!=i:
              self.swap(heap, maxi, i)
              self.heapify(heap, maxi, n)
  
      def sort(self, heap):
          self.build_heap(heap, len(heap))
          end = len(heap)-1
          while end > 0:
              self.swap(heap, 0, end)
              self.heapify(heap, 0, end)
              end -= 1
              
  
      def maxProduct(self, nums: List[int]) -> int:
          self.sort(nums)
          print(nums)
          return (nums[-1]-1)*(nums[-2]-1)
  ```

  
