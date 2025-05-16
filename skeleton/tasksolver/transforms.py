from arctypes import *

def identity(x):
    """Identity function."""
    return x

def add(x: Numerical, y: Numerical) -> Numerical:
    """Add two numbers."""
    if isinstance(x, int) and isinstance(y, int):
        return x + y
    if isinstance(x, tuple) and isinstance(y, tuple):
        return (x[0] + y[0], x[1] + y[1])
    if isinstance(x, int) and isinstance(y, tuple):
        return (x + y[0], x + y[1])
    return (x[0] + y, x[1] + y)

def sub(x: Numerical, y: Numerical) -> Numerical:
    """Subtract two numbers."""
    if isinstance(x, int) and isinstance(y, int):
        return x - y
    if isinstance(x, tuple) and isinstance(y, tuple):
        return (x[0] - y[0], x[1] - y[1])
    if isinstance(x, int) and isinstance(y, tuple):
        return (x - y[0], x - y[1])
    return (x[0] - y, x[1] - y)

def mul(x: Numerical, y: Numerical) -> Numerical:
    """Multiply two numbers."""
    if isinstance(x, int) and isinstance(y, int):
        return x * y
    if isinstance(x, tuple) and isinstance(y, tuple):
        return (x[0] * y[0], x[1] * y[1])
    if isinstance(x, int) and isinstance(y, tuple):
        return (x * y[0], x * y[1])
    return (x[0] * y, x[1] * y)

def div(x: Numerical, y: Numerical) -> Numerical:
    """Divide two numbers."""
    if isinstance(x, int) and isinstance(y, int):
        return x // y
    if isinstance(x, tuple) and isinstance(y, tuple):
        return (x[0] // y[0], x[1] // y[1])
    if isinstance(x, int) and isinstance(y, tuple):
        return (x // y[0], x // y[1])
    return (x[0] // y, x[1] // y)

def inv(x: Numerical) -> Numerical:
    """Invert a number."""
    if isinstance(x, int):
        return -x
    return (-x[0], -x[1])

def is_even(x: Numerical) -> bool:
    """Check if a number is even."""
    if isinstance(x, int):
        return x % 2 == 0
    return x[0] % 2 == 0 and x[1] % 2 == 0

def double(x: Numerical) -> Numerical:
    """Double a number."""
    if isinstance(x, int):
        return x * 2
    return (x[0] * 2, x[1] * 2)

def halve(x: Numerical) -> Numerical:
    """Halve a number."""
    if isinstance(x, int):
        return x // 2
    return (x[0] // 2, x[1] // 2)

def is_equal(x: Any, y: Any) -> bool:
    """Check if two values are equal."""
    return x == y

def contains(x: Container, y: Any) -> bool:
    """Check if a container contains a value."""
    return y in x

def combine(x: Container, y: Container) -> Container:
    """Combine two containers."""
    return type(x)(*x, *y)

def intersect(x: Container, y: Container) -> Container:
    """Intersect two containers."""
    return x & y

def difference(x: Container, y: Container) -> Container:
    """Difference of two containers."""
    return type(x)(e for e in x if e not in y)

def dedupe(tup: Tuple) -> Tuple:
    """Remove duplicates from a tuple."""
    return tuple(e for i, e in enumerate(tup) if tup.index(e) == i)

def order(con: Container, comp: Callable) -> Container:
    """Order a container using a comparison function."""
    return tuple(sorted(con, key=comp))

def repeat(x: Any, n: int) -> Container:
    """Repeat a value n times."""
    return tuple(x for _ in range(n))

def greater(x: int, y: int) -> bool:
    """Check if x is greater than y."""
    return x > y

def size(con: Container) -> int:
    """Get the size of a container."""
    return len(con)

def merge(cons: ContainerContainer) -> Container:
    """Merge multiple containers."""
    return type(cons)(e for con in cons for e in con)

def maxim(intset: IntegerSet) -> int:
    """Get the maximum value from a set of integers."""
    return max(intset, default=0)

def minim(intset: IntegerSet) -> int:
    """Get the minimum value from a set of integers."""
    return min(intset, default=0)

def compmax(con: Container, comp: Callable) -> int:
    """Get the maximum value from a set of integers using a comparison function."""
    return comp(max(con, key=comp, default=0))

def compmin(con: Container, comp: Callable) -> int:
    """Get the minimum value from a set of integers using a comparison function."""
    return comp(min(con, key=comp, default=0))

def argmax(con: Container, comp: Callable) -> int:
    """Get the index of the maximum value from a set of integers using a comparison function."""
    return max(con, key=comp)

def argmin(con: Container, comp: Callable) -> int:
    """Get the index of the minimum value from a set of integers using a comparison function."""
    return min(con, key=comp)

def mostfreq(con: Container) -> Any:
    """Get the most frequent value from a container."""
    return max(set(con), key=con.count)

def leastfreq(con: Container) -> Any:
    """Get the least frequent value from a container."""
    return min(set(con), key=con.count)

def initset(x: Any) -> FrozenSet:
    """Initialize a set with a single value."""
    return frozenset({x})

def flip(x: Boolean) -> bool:
    """Flip a boolean value."""
    return not x

def both(x: Boolean, y: Boolean) -> bool:
    """Check if both values are true."""
    return x and y

def either(x: Boolean, y: Boolean) -> bool:
    """Check if either value is true."""
    return x or y

def successor(x: Numerical) -> Numerical:
    """Get the successor of a number."""
    if isinstance(x, int):
        return x + 1
    return (x[0] + 1, x[1] + 1)

def predecessor(x: Numerical) -> Numerical:
    """Get the predecessor of a number."""
    if isinstance(x, int):
        return x - 1
    return (x[0] - 1, x[1] - 1)

def spread(x: Numerical) -> Numerical:
    """Incrementing positive and decrementing negative."""
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
            0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1))

def sign(x: Numerical) -> int:
    """Get the sign of a number."""
    if isinstance(x, int):
        return 1 if x > 0 else -1 if x < 0 else 0
    return (1 if x[0] > 0 else -1 if x[0] < 0 else 0, 
            1 if x[1] > 0 else -1 if x[1] < 0 else 0)

def ispositive(x: Numerical) -> bool:
    """Check if a number is positive."""
    if isinstance(x, int):
        return x > 0
    return x[0] > 0 and x[1] > 0

def toivec(x: Integer) -> IntegerTuple:
    """Convert an integer to a vector pointing vertically."""
    return (x, 0)

def tojvec(x: Integer) -> IntegerTuple:
    """Convert an integer to a vector pointing horizontally."""
    return (0, x)

def sfilter(con: Container, cond: Callable) -> Container:
    """Filter a container using a condition."""
    return type(con)(e for e in con if cond(e))

def mfilter(con: Container, cond: Callable) -> FrozenSet:
    """Filter and merge."""
    return merge(sfilter(con, cond))

def extract(con: Container, cond: Callable) -> Any:
    """Extract a value from a container using a condition."""
    return next(e for e in con if cond(e))

def totuple(con: FrozenSet) -> Tuple:
    """Convert a frozen set to a tuple."""
    return tuple(con)

def first(con: Container) -> Any:
    """Get the first element of a container."""
    return next(iter(con))

def last(con: Container) -> Any:
    """Get the last element of a container."""
    return max(enumerate(con))[1]

def insert(con: FrozenSet, x: Any) -> FrozenSet:
    """Insert an element into a frozen set."""
    return con.union(frozenset({x}))

def remove(con: FrozenSet, x: Any) -> FrozenSet:
    """Remove an element from a frozen set."""
    return type(con)(e for e in con if e != x)

def other(con: Container, x: Any) -> Any:
    """Get the other element from a container."""
    return first(remove(con, x))

def interval(begin: Integer, end: Integer, step: Integer) -> Container:
    """Create an interval from start to end with a step."""
    return tuple(range(begin, end, step))

def astuple(x: Integer, y: Integer) -> IntegerTuple:
    """Convert two integers to a tuple."""
    return (x, y)

def product(x: Container, y: Container) -> FrozenSet:
    """Get the Cartesian product of two containers."""
    return frozenset((a, b) for a in x for b in y)

# line 423