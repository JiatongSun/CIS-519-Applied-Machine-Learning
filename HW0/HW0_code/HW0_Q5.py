import numpy as np

def CompletePath(s, w, h) -> str:
    '''This function is used to escape from a room whose size is w * h.
    You are trapped in the bottom-left corner and need to cross to the
    door in the upper-right corner to escape.
    @:param s: a string representing the partial path composed of {'U', 'D', 'L', 'R', '?'}
    @:param w: an integer representing the room width
    @:param h: an integer representing the room length
    @:return path: a string that represents the completed path, with all question marks in with the correct directions
    or None if there is no path possible.
    '''
    
    # # TODO # #
    
    # error-check
    assert isinstance(s,str), "s is not a string!"
    assert isinstance(w,int), "w is not an integer!"
    assert isinstance(h,int), "h is not an integer!"
    
    # function
    ori = {'U':(0,1),'D':(0,-1),'L':(-1,0),'R':(1,0),"?":(0,0)}
    choice = "UDLR"
    start = (0,0)                               # the start coordinate
    y, x = start
    used = np.zeros((h, w), dtype = np.int32)   # document passed coordinates
    used[y][x] = 1                              # the start coordinate is passed
    idx = 0                                     # the index of the string
    for i in s:
        dx,dy = ori[i]
        if not(dx==0 and dy==0):                # the path is not "?"
            x, y = x + dx, y + dy               # new coordinate
            # return if out of range
            if x<0 or x>w-1 or y<0 or y>h-1 or used[y][x]==1:
                return
            else:
                used[y][x] = 1                  # current coordinate used
                
            # judge the result path
            if x == w-1 and y == h-1:
                if idx == len(s)-1:
                    return s
                else:
                    continue
            elif idx == len(s)-1:
                continue
            else:
                idx = idx + 1                   # continue the path
        else:                                   # the path is "?"
            # enumerate 4 directions
            for j in choice:
                try_s = s[:idx] + j + s[idx+1:] # guess the first "?"
                sol = CompletePath(try_s,w,h)   # solve the path based on guess
                if sol:
                    return sol
                else:
                    continue

#s=CompletePath("?RDRR?UUUR", 5, 5)
#s=CompletePath("UURDD?UUR?RR", 6, 4)
#s=CompletePath("????????", 3, 3)
#print(s)