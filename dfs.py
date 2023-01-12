"""
https://ebisuke33.hatenablog.com/entry/antbook-aoj1160
"""
import sys
sys.setrecursionlimit(10**7) #再帰回数の上限変更

def how_many_island(img):
    # マスy,xに隣接する陸を海に置き換え
    def dfs(y,x):
        # 現在地を0に置き換え
        img[y][x] = 0
        # 周囲8マスをループ
        for dx in range(-1,2):
            for dy in range(-1,2):
                # 移動後のマスをny,nxとする
                ny = y + dy
                nx = x + dx

                # ny,nxがfield内で陸かどうかを判別
                if 0 <= nx < w and 0 <= ny < h and img[ny][nx] == 1:
                    dfs(ny,nx)
        return

    w, h = len(img[0]), len(img)

    # 島の数
    ans = 0

    for i in range(h):
        for j in range(w):
            if img[i][j] == 1:
                dfs(i,j)
                ans += 1

    return ans