# leetcode简单题

[TOC]




## 1. 字符串反转

反转一个字符串

- StringBuilder

  ``` java
  class Solution {
      public String reverse(String x){
          return new StringBuilder(x).reverse().toString();
      }
  }
  ```

- 字符数组

  ``` java
  class Solution {
      public String reverse(String x){
          char[] arr = x.toCharArray();
          for (int i = 0; i < x.length/2; i++){
              char temp = arr[i];
              arr[i] = arr[arr.length-i-1];
              arr[arr.length-i-1] = temp;
          }
          return new String(arr)
          // return String.valueOf(arr)
      }
  }
  ```

- charAt

  ``` java
  class Solution {
      public String reverse(String x){
          String ans = "";
          for (int i = x.length()-1; i >= 0; i--){
              ans += x.charAt(i);
          }
          return ans;
      }
  }
  ```

## 2. [回文数](https://leetcode-cn.com/problems/palindrome-number/)

判断一个整数是不是回文数

* 字符串解法

  ``` java
  class Solution {
      public boolean isSymmetry(int x){
          String v = Integer.toString(x);
          for (int i = 0; i < v.length(); i++){
              if (v.charAt(i)!=v.charAt(v.length()-1-i))
                  return false;
          }
          return true;
      }
  }
  
  // 字符数组
  class Solution {
      public boolean isSymmetry(int x){
          char[] v = Integer.toString(x).toCharArray();
  		for (int i = 0; i < v.length; i++){
              if (v[i]!=v[v.length-1-i])
                  return false;
          }
          return true;
      }
  }
  
  // 反转
  class Solution {
      public boolean isSymmetry(int x){
          String v = Integer.toString(x);
         	String rev = new StringBuilder(v).reverse().toString();
          return v.equals(rev);
      }
  }
  ```

* 数值解法

  求出整数x的反转整数（取模，除10），比较是否与x相等

  ``` java
  class Solution {
      public boolean isSymmetry(int x){
  		if (x < 0)
              return false;
          
          int rev = 0;
          final int temp = x;
          
          while (x > 0){
              rev = rev*10 + x%10;
              x /= 10;
          }
          return rev==temp;
      }
  }
  ```

## 3. [整数反转](https://leetcode-cn.com/problems/reverse-integer/)

整数反转数值解法：

1. 原数字x对10取模得到余数pop，然后整除10
2. 新数字rev乘10再加上余数pop (rev初始为0)
3. 重复1、2

这道题的难点在于数字有范围限制，需要判断数字溢出的情况。以8位有符号数字[-128, 127]为例，考虑临界条件：

- 若rev > 12或rev < -12，执行第2步后，数字必定溢出。
- 若rev=12，且pop > 7，则溢出。
- 若rev=-12，且pop < -8，则溢出。

此处，7=127%10，-8=-128%10。
对于32位有符号数字[-2^31^, -2^31^-1]，-2^31^-1的二进制末尾3位111，因此对10取模为7，2^31^ 的二进制末尾为000，因此-2^31^对10取模为-8.

``` java
class Solution {
    public int reverse(int x) {
        int INT_MAX = (int)(Math.pow(2, 31)-1);
        int INT_MIN = (int)(-Math.pow(2, 31));

        int rev = 0;

        while (x != 0) {
            // 临界条件1
            if (rev > INT_MAX/10 || rev < INT_MIN/10) 
                return 0;

            int pop = x % 10;

            // 临界条件2
            if (rev==INT_MAX/10 && pop > 7 || rev==INT_MIN/10 && pop < -8)
                return 0;

            rev = rev*10 + pop;
            x /= 10;
        }

        return rev;
    }
}
```


## 4. [两数之和](https://leetcode-cn.com/problems/two-sum/)

两次遍历

- 遍历数组，使用一个hashmap记录每个元素的值和索引
- 再次遍历数组，每遍历一个元素x，就判断target-x是否在hashmap中:
  - 如果存在，且x的索引!=hashmap[target-x]，则返回[x的索引，hashmap[target-x]]，结束
  - 否则，继续遍历

``` java
class Solution {
    public int[] twoSum(int[] nums, int target) {
		Map<Integer, Integer> hashmap = new HashMap<>();
		
		for (int i = 0; i < nums.length; i++) {
			hashmap.put(nums[i], i);
		}
		
		for (int i = 0; i < nums.length; i++) {
			if (hashmap.containsKey(target-nums[i]) && hashmap.get(target-nums[i])!=i)
				return new int[] {hashmap.get(target-nums[i]), i};
		}
		return null;
    }
}
```

一次遍历

- 使用一
- 个hashmap用于记录元素的值和索引
- 遍历数组，每遍历一个元素x，就看target-x是否在hashmap中
  - 如果存在，则返回[x的索引，hashmap[target-x]]，结束
  - 如果不存在，就将[x, x的索引]存入hashmap中，继续遍历

``` java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hashmap = new HashMap<>();      
        for (int i = 0; i < nums.length; i++) {
            if (hashmap.containsKey(target-nums[i]))
                return new int[] {hashmap.get(target-nums[i]), i};
            else
                hashmap.put(nums[i], i);
        }
        return null;
    }
}
```

## 5. [罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

`I`， `V`， `X`， `L`，`C`，`D` 和 `M`

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

设置初值sum，遍历字符串，如果当前字符比后一个字符对应的数值小，就作减法，否则就作加法。

``` java
class Solution {
	public int romanToInt(String s) {
		Map<Character, Integer> map = new HashMap<>();
		map.put('I', 1);
		map.put('V', 5);
		map.put('X', 10);
		map.put('L', 50);
		map.put('C', 100);
		map.put('D', 500);
		map.put('M', 1000);

		int sum = 0;
		int prev = map.get(s.charAt(0));
		for (int i = 1; i < s.length(); i++) {
			if (prev < map.get(s.charAt(i)))
				sum -= prev;
			else
				sum += prev
		}
		return sum;
	}
}
```

## 6. [有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

遍历括号字符串，考虑入栈和出栈的情况：

- 栈空或当前字符为左括号，则入栈
- 当前字符和栈顶字符匹配，则出栈

程序出口条件

- 遍历字符串的过程中，既不满足出栈的条件，又不满足入栈的条件，返回false
- 遍历结束后，栈为空则返回true，否则为false

``` java
class Solution {
	public boolean isValid(String s) {
		Stack<Character> stk = new Stack<>();
		for (int i = 0; i < s.length(); i++) {
            if (stk.isEmpty() || s.charAt(i)=='(' || s.charAt(i)=='{' 
            || s.charAt(i)=='[') {
                stk.push(s.charAt(i));
                continue;
            }

			char top = stk.peek();
			if (s.charAt(i)==')' && top=='(' || s.charAt(i)==']' && top=='[' ||s.charAt(i)=='}' && top=='{') {
                stk.pop();
            } else
				return false;		
		}
		return stk.isEmpty();
	}
}
```

使用hashmap优化

``` java
class Solution {
	public boolean isValid(String s) {
		Map<Character, Character> map = new HashMap<Character, Character>(){{
			put('(', ')');
			put('[', ']');
			put('{', '}');
		}};
		Stack<Character> stk = new Stack<>();
		
		for (int i = 0; i < s.length(); i++) {
			if (stk.isEmpty() || s.charAt(i)=='(' || s.charAt(i)=='[' || s.charAt(i)=='{') {
				stk.push(s.charAt(i));
				continue;
			}

			char top = stk.peek();
			if (map.get(top)==s.charAt(i)) {
				stk.pop();
			} else {
				return false;
			}
		}
		
		return stk.isEmpty();
	}
}
```

## 7. [合并两个有序链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

非递归解法

1. 创建临时节点node和尾指针tail
2. 两个链表分别由一个指针p和q指向
3. 判断p和q对应的值
   - 若p.val < q.val则tail.next = p, p = p.next, tail = tail.next
   - 否则，tail.next = q, q = q.next, tail = tail.next
4. 重复3，直至p和q有一方为null
5. tail.next = p和q不空的一方

``` java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode node = new ListNode(0);
        ListNode p = node;
        while (l1!=null && l2!=null) {
            if (l1.val < l2.val) {
                p.next = l1;
                l1 = l1.next;
                p = p.next;
            } else {
                p.next = l2;
                l2 = l2.next;
                p = p.next;
            }
        }

        p.next = l1!=null ? l1 : l2;

        return node.next;
    }
}
```

递归解法

- 边界条件: p和q有一方为null，返回另一方
- 比较p.val和q.val
  -若p.val < q.val，则p.next = merge(p.next, q)，return p;
  -否则，q.next = merge(p, q.next), return q;

``` java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1==null)
            return l2;
        if (l2==null)
            return l1;
        
        if (l1.val < l2.val){
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
            
    }
}
```

## 8. [删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

双指针

- 慢指针i填数字
- 快指针j找不重复的项
- 初始条件 i = j = 0

``` java
class Solution {
    public int removeDuplicates(int[] nums) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[i]!=nums[j]){
                i++;
                nums[i] = nums[j];
            }
        }
        return i+1;
    }
}
```

## 9. [移除元素](https://leetcode-cn.com/problems/remove-element/)

题目：原地移除指定值val。

无论是移除重复元素还是移除指定元素，都是把保留元素放在一起，因此可以使用双指针。

双指针法

- 慢指针填数字
- 快指针找不等于val的值

注意：移除元素和移除重复元素，在慢指针填数和返回值上有些差异。

``` java
class Solution {
    public int removeElement(int[] nums, int val) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j]!=val){
                nums[i] = nums[j];
                i++;
            }
        }
        return i;
    }
}
```

## 10. [搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

二分查找

- 循环判断条件 left <= right
- 临界条件
  - a[mid]==val，返回mid
  - left==right，返回right

``` java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid]==target)
                return mid;
            else if (nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return left;
    }
}
```

## 11. 最长公共前缀

横向扫描

先计算两个字符串的公共前缀，再与第三个字符串计算公共前缀。
注意当字符串数组长度为0时，返回空串，当字符串数组长度为1时，返回数组第一个元素。

- 初始化prefix = strs[0]
- 计算prefix与第二个字符串的公共前缀，更新prefix
- 继续与后面的字符串计算公共前缀并更新。

``` java
class Solution {

    public String common(String s1, String s2) {
        int i = 0;
        while (i < s1.length() && i < s2.length() && s1.charAt(i)==s2.charAt(i))
            i++;
        return s1.substring(0, i);
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs.length==0)
            return "";
        
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            prefix = common(prefix, strs[i]);
        }
        return prefix;
    }
}
```

## 12. 两数之和II 输入有序数组

双指针法

头指针和尾指针分别向中间移动，两数之和比目标值小，则头指针移动，两数之和比目标值大，则尾指针移动。

``` java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int i = 0;
		int j = numbers.length-1;
		while (i < j) {
			if (numbers[i] + numbers[j] == target)
				return new int[] {i+1, j+1};
			else if (numbers[i] + numbers[j] < target)
				i++;
			else
				j--;
		}
		return null;
    }
}
```

## 13. 删除排序链表中的重复元素

- 使用一个指针p遍历链表
- 每遍历到一个节点，就判断该节点值和下一个节点值
  - 如果相等，则删除下一个节点
  - 否则，指针指向下一个节点

``` java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head==null)
            return head;
            
        ListNode p = head;
        while (p.next!=null) {
            if (p.next.val==p.val){
                p.next = p.next.next;
            } else
                p = p.next;
        }
        return head;
    }
}
```

## 14. 重复的子字符串

暴力解法

暴力解法需要考虑两个问题

- 字串长度选择
- 重复子字符串判断

字串长度有两个选择条件

- 字串长度不超过字符串长度的一半
- 字串长度应该能被字符串长度整除

重复子字符串判断，需要利用重复字符串的性质:
i 是重复子串的长度，则对于j >= i，有s[j] = s[j-i]

``` java
class Solution {
    public boolean repeatedSubstringPattern(String s) {
        for (int i = 1; i <= s.length()/2; i++) {
            if (s.length()%i != 0)
                continue;
            
            boolean isRepeated = true;

            for (int j = i; j < s.length(); j++) {
                if (s.charAt(j)!=s.charAt(j-i)){
                    isRepeated = false;
                    break;
                }
            }

            if (isRepeated)
                return isRepeated;
        }

        return false;
    }
}
```

双倍字符串解法

拼接两个字符串s，去除首尾字符，判断剩余字符串是否包含s

``` java
class Solution {
    public boolean repeatedSubstringPattern(String s) {
        return (s + s).indexOf(s, 1) != s.length();
    }
}

// 稍慢
class Solution {
    public boolean repeatedSubstringPattern(String s) {
        return (s + s).substring(1, 2*s.length()-1).contains(s);
    }
}
```

## 15. 约瑟夫环

m个人围成一个环，第一个人从1开始报数，报到n的人出列。从出列的下一个人开始从1报数，报到n的人出列......

数组实现
for循环 + while循环

- for(int i = 0; i < m-1; i++)：出列m-1次
- while(count > 0) 报数

``` java
class Solution {
	public int joseph(int m, int n) {
		int[] arr = new int[m];
		for (int i = 0; i < arr.length; i++) 
			arr[i] = i + 1;
		
		int j = 0;
		for (int i = 0; i < arr.length-1; i++) {
			int count = n;
			while (count > 0) {
				if (arr[j]!=-1)
					count--;
				if (count==0)
					arr[j]=-1;
				j = (j + 1)%arr.length;
			}
		}
		
		for (int i = 0; i < arr.length; i++)
			if (arr[i]!=-1)
				return arr[i];
		return -1;
	}
}
```

递推公式
f(n, m) = (f(n-1, m) + m)%n
注：如果从1开始报数，调用joseph函数后应该再 + 1

``` java
class Solution {
	public int joseph(int m, int n) {
		if (m==0 || n==0)
			return -1;
		return (joseph(m-1, n) + n) % m;
	}
}
```



## 16. 旋转数组

每次右移一位（原地）

- k要先对数组长度取模
- 右移一位：先记录最后一个元素，然后把除最后一个元素的所有元素右移一位，最后把最后一个元素放在第一个位置。

``` java
class Solution {
    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        for (int i = 0; i < k; i++) {
            int last = nums[nums.length-1];

            for (int j = nums.length-1; j > 0; j--) {
                nums[j] = nums[j-1];
            }

            nums[0] = last;
        }
    }
}
```

反转（原地）

- k对数组长度取模
- 反转整个数组
- 反转前k个元素
- 反转后半部分元素

``` java
class Solution {
   	public void rotate(int[] nums, int k) {
        k = k % nums.length;
		reverse(nums, 0, nums.length-1);
		reverse(nums, 0, k-1);
		reverse(nums, k, nums.length-1);
	}
	
	public void reverse(int[] nums, int left, int right) {
		while (left < right) {
			int temp = nums[left];
			nums[left] = nums[right];
			nums[right] = temp;
			left++;
			right--;
		}
	}
}
```

环状替换

``` java
public class Solution {
    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        int count = 0;
        for (int start = 0; count < nums.length; start++) {
            int current = start;
            int prev = nums[start];
            do {
                // 注意 此处不是交换
                int next = (current + k) % nums.length;
                int temp = nums[next];
                nums[next] = prev;
                prev = temp;
                current = next;
                count++;
            } while (start != current);
        }
    }
}
```

## 17. 第一个错误的版本

二分法

这道题可以看作搜索插入位置的特殊情况：在已经排序的数组中（true = 0, false=1），找到第一个false(1)的插入位置.

``` java
public class Solution extends VersionControl {
  public int firstBadVersion(int n) {
      int left = 1;
      int right = n;
      while (left <= right) {
          int mid = left + (right - left) / 2;
          if (isBadVersion(mid)) {
              right = mid - 1;
          } else {
              left = mid + 1;
          }
      }
      return left;
      
  }
}
```



## 18. 完全平方数

二分法

如果使用int类型，会超时，把int改成long即可。

``` java
class Solution {
  public boolean isPerfectSquare(int num) {
      if (num < 2)
        return true;

      long left = 1;
      long right = num/2;
      long val;

      while (left <= right) {
          long mid = left + (right-left)/2;
          val = mid*mid;
          if (val==num)
            return true;
          else if (val < num)
            left = mid + 1;
          else
            right = mid - 1;
      }

      return false;
  }
}
```

完全平方数的性质

任何一个完全平方数都可以写成从1开始的奇数的累加和。

``` java
class Solution {
  public boolean isPerfectSquare(int num) {
      int i = 1;
      while (i < num) {
          num -= i;
          i += 2;
      }

      return i==num;
  }
}
```



## 19. 完美数

``` java
class Solution {
    public boolean checkPerfectNumber(int num) {
        int sum = 0;

        for (int i = 2; i*i <= num; i++) {
            if (num%i==0) {
                sum += i;
                sum += num/i;
            }
        }

        int k = (int) Math.sqrt(num);
        if (k*k==num)
            sum -= k;

        sum++;

        return sum==num;
    }
}
```



## [20. 子数组的最大平均数](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

滑动窗口法

计算第一个窗口内的元素和，向右滑动，加上右边的元素，减去窗口第一个元素。

``` java
class Solution {
    public double findMaxAverage(int[] nums, int k) {
        double sum = 0;
        for (int i = 0; i < k; i++)
            sum += nums[i];

        double maxSum = sum;
        
        for (int i = k; i < nums.length; i++) {
            sum += nums[i] - nums[i-k];
            maxSum = maxSum < sum ? sum : maxSum;
        }

        return maxSum/k;
    }
}
```

## [21. 平方数之和]([https://leetcode-cn.com/problems/sum-of-square-numbers/](https://leetcode-cn.com/problems/sum-of-square-numbers/))

遍历

- 遍历区间 [0, sqrt(c)]

``` java
class Solution {
    public boolean judgeSquareSum(int c) {
        int end = (int) Math.sqrt(c);
        for (int a = 0; a <= end; a++) {
            int temp = c - a*a;
            if (Math.pow((int) Math.sqrt(temp), 2)==temp)
                return true;
        }

        return false;

    }
}
```

双指针

类似于两数之和II 输入有序数组。

``` java
class Solution {
    public boolean judgeSquareSum(int c) {
        int left = 0;
        int right = (int) Math.sqrt(c);
        
        while (left <= right) {
            if (left*left + right*right==c)
                return true;
            else if (left*left + right*right < c)
                left++;
            else
                right--;
        }

        return false;

    }
```

## 22. [三个数最大乘积]([https://leetcode-cn.com/problems/maximum-product-of-three-numbers/](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/))

排序

最大乘积来源

- 最大的三个正数(+++)
- 最小的两个负数和最大的正数(--+)

``` java
class Solution {
    public int maximumProduct(int[] nums) {
        if (nums.length < 3)
            return -1;

        Arrays.sort(nums);
        int max1 = nums[0]*nums[1]*nums[nums.length-1];
        int max2 = nums[nums.length-1]*nums[nums.length-2]*nums[nums.length-3];

        return max1 > max2 ? max1 : max2;

    }
}
```

线性扫描

不排序，通过扫描直接找出最大的三个整数和最小的两个负数。
技巧：同时得到最大的三个值和最小的两个值。

``` java
class Solution {
    public int maximumProduct(int[] nums) {
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;

        for (int val: nums) {
            if (val > max1) {
                max3 = max2;
                max2 = max1;
                max1 = val;
            } else if (val > max2) {
                max3 = max2;
                max2 = val;
            } else if (val > max3) {
                max3 = val;
            } 

            if (val < min1) {
                min2 = min1;
                min1 = val;
            } else if (val < min2) {
                min2 = val;
            }  
        }

        return Math.max(min1*min2*max1, max1*max2*max3);

    }
}
```

## 23. [分糖果]([https://leetcode-cn.com/problems/distribute-candies/](https://leetcode-cn.com/problems/distribute-candies/))

集合

统计糖果类别，返回min(糖果类别个数，糖果个数的一半)

``` java
class Solution {
    public int distributeCandies(int[] candies) {
        Set<Integer> cds = new HashSet<>();
        for (int cand: candies) {
            cds.add(cand);
        }

        return Math.min(cds.size(), candies.length/2);
    }
}
```

排序

``` java
class Solution {
    public int distributeCandies(int[] candies) {
        Arrays.sort(candies);
        int candy = candies[0];
        int count = 1;
        for (int i = 1; i < candies.length; i++) {
            if (candy!=candies[i]) {
                count++;
                candy = candies[i];
            }
        }
        return Math.min(count, candies.length/2);
    }
}
```

## 24. [范围求和II](https://leetcode-cn.com/problems/range-addition-ii/)


求交集
统计最大值的个数，本质是在求交集。

``` java
class Solution {
    public int maxCount(int m, int n, int[][] ops) {
        if (ops.length==0)
            return m*n;
            
        int minx = ops[0][0];
        int miny = ops[0][1];

        for (int[] op: ops) {
            if (minx > op[0])
                minx = op[0];
            if (miny > op[1])
                miny = op[1];
        }

        return minx*miny;
    }
}
```


## 22. 最长和谐子序列

计数法
使用hashmap统计每个数字出现的次数，然后遍历hashmap的key和value，记录max(hashmap[key], hashmap[key+1])

``` java
import java.util.*;
class Solution {
    public int findLHS(int[] nums) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int num: nums) {
            if (counts.containsKey(num)) {
                counts.put(num, counts.get(num) + 1);
            } else {
                counts.put(num, 1);
            }
        }

        int ans = 0;
        for (Integer key: counts.keySet()) {
            if (counts.containsKey(key + 1)) {
                ans = Math.max(ans, counts.get(key) + counts.get(key+1));
            }
        }

        return ans;

    }
}
```


## 23. 图片平滑器

偏移数组
定义偏移数组dx和dy，长度为8。表示相对当前坐标的偏移位置，借助偏移数组能够访问到当前坐标的所有邻居，求和取平均。

``` java
class Solution {
    public int[][] imageSmoother(int[][] M) {
        int m = M.length;
        int n = M[0].length;
        int[][] mat = new int[m][n];

        int[] dx = {1, 1, 0, -1, -1, -1, 0, 1};
        int[] dy = {0, -1, -1, -1, 0, 1, 1, 1};

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int count = 1;
                int sum = M[i][j];
                for (int idx = 0; idx < 8; idx++) {
                    int ni = i + dx[idx];
                    int nj = j + dy[idx];
                    
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n){
                        count++;
                        sum += M[ni][nj];
                    }
                }

                mat[i][j] = sum/count;

            }
        }

        return mat;
    }
```

## 24. 种花问题

贪心算法

``` java
class Solution {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int count = 0;
        for (int i = 0; i < flowerbed.length; i++) {
            if (flowerbed[i]==0 && 
            (i==flowerbed.length-1 || flowerbed[i+1]==0) && 
            (i==0 || flowerbed[i-1]==0)) {
                count++;
                flowerbed[i] = 1;
            }
        }
        return count >= n;
    }
}
```

## 25. 错误的集合

集合

``` java
class Solution {
    public int[] findErrorNums(int[] nums) {
        int sum = 0;
        for (int num: nums)
            sum += num;

        Set<Integer> set = new HashSet<>();

        int dup = 0;
        for (Integer num: nums) {
            if (set.contains(num)) {
                dup = num;
                break;
            } else {
                set.add(num);
            }
        }

        sum -= dup;

        int lost = (1 + nums.length)*nums.length/2 - sum;

        return new int[] {dup, lost};

    }
```