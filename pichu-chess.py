#!/usr/bin/env python3
# pichu.py : Solve the chessboard!

"""

- Initial code check-in, we could easily get the moves of all chess pieces, successor function and evaluation function
  wrote the minimax algorithm which went into an infinite loop, we had to check where it was going wrong.

- We took time to get the evaluation function right, we got the minimax to work and we then worked on alpha beta pruning algorithm.

- We have used evaluation function described in the site - https://chessprogramming.wikispaces.com/Simplified+evaluation+function
  for computing the best possible next move from the given state.

- Some issues that need to be addressed, during some moves the alpha beta pruning algorithm is taking longer than 5 sec which is
  causing our program to return no moves and forfeit our turn and there by causing our program to quit.
  This will need to reworked and is an interesting challenge.

"""
import sys
import signal
import math
import timeit
from copy import deepcopy
import time


# assigned a maximum depth of 3 for the min max algorithm to consider
MAX_DEPTH = 2


# function used to get the desired format of the chess board
def append(board):
    str_1 = ''
    for i in range(0, 8):
        for j in range(0, 8):
            str_1 = str_1 + board[i][j]
    return str_1

def player_pieces(player):
    if player=='w':
        return ('R', 'N', 'B', 'Q', 'K', 'P')
    return ('r', 'n', 'b', 'q', 'k', 'p')

def player_take_out(player):
    if player=='w':
        return ('r', 'n', 'b', 'q', 'k', 'p')
    return ('R', 'N', 'B', 'Q', 'K', 'P')

# changing the given chess board into a 8x8 matrix
def initial_board(board):
    curr_board = [['.'] * 8 for _ in range(8)]

    list_board = []

    for ltr in board:
        list_board.append(ltr)
    k = 0
    for i in range(0, 8):
        for j in range(0, 8):
            curr_board[i][j] = list_board[k]
            k = k + 1
    return curr_board


# for printing the 8x8 matrix board
def printable_board(board):
    return print("\n".join([" ".join([board[row][col] for col in range(0, 8)]) for row in range(0, 8)]))


# evaluation function used in the initial stage of the program
def evaluation_d(board):
    king, queen, rook, bishop, knight, pawn = 900, 90, 50, 30, 30, 10
    cnt_max_pawn, cnt_max_rook, cnt_max_knight, cnt_max_bishop, cnt_max_queen, cnt_max_king = count(board, MAX)
    cnt_min_pawn, cnt_min_rook, cnt_min_knight, cnt_min_bishop, cnt_min_queen, cnt_min_king = count(board, MIN)
    weighted_sum = (cnt_max_pawn - cnt_min_pawn) * pawn + (cnt_max_rook - cnt_min_rook) * rook + \
                   (cnt_max_knight - cnt_min_knight) * knight + (cnt_max_bishop - cnt_min_bishop) * bishop + \
                   (cnt_max_queen - cnt_min_queen) * queen + (cnt_max_king - cnt_min_king) * king
    return weighted_sum


# function to count each kind of piece in the chessboard
def count(board, player):
    count_pawn, count_rook, count_knight, count_bishop, count_queen, count_king = 0, 0, 0, 0, 0, 0
    for ltr in board:
        if ltr != '.':
            if ltr == parakeet:
                count_pawn = count_pawn + 1
            if ltr == robin:
                count_rook = count_rook + 1
            if ltr == nighthawk:
                count_knight = count_knight + 1
            if ltr == bluejay:
                count_bishop = count_bishop + 1
            if ltr == quetzal:
                count_queen = count_queen + 1
            if ltr == kingfisher:
                count_king = count_king + 1

    return count_pawn, count_rook, count_knight, count_bishop, count_queen, count_king


# gets the current position of each kind of piece in the chessboard
def curr_position(board, piece):
    # planning to get the position of a piece and append it to the possible moves.
    position = list()
    for row in range(0, 8):
        for column in range(0, 8):
            if board[row][column] == piece:
                position.append([row, column])
    return position


# Gets all possible Robin Moves
def robin_moves(player, position, board):
    row, column = position[0], position[1]
    allmoves = []

    piece = 'R' if player == 'w' else 'r'

    takeout_pieces = player_take_out(player)
    pieces = player_pieces(player)

    # the moves in row
    for j in range(0, 8):
        if j != column:
            if board[row][j] in takeout_pieces:
                allmoves.append([piece, row, j, row, column])
                break
            elif board[row][j] in pieces:
                break
            else:
                allmoves.append([piece, row, j, row, column])

    # the moves in column
    for i in range(0, 8):
        if i != row:
            if board[i][column] in takeout_pieces:
                allmoves.append([piece, i, column, row, column])
                break
            elif board[i][column] in pieces:
                break
            else:
                allmoves.append([piece, i, column, row, column])

    allmoves.sort()

    return allmoves


# Gets all possible Nighthawk moves
def nighthawk_moves(player, position, board):
    row, column = position[0], position[1]
    i, j = row, column
    moves = []
    piece = 'N' if player == 'w' else 'n'
    pieces = player_pieces(player)

    if i + 1 < 8 and j - 2 >= 0 and board[i + 1][j - 2] not in pieces:
        moves.append([piece, i + 1, j - 2, row, column])

    if i + 2 < 8 and j - 1 >= 0 and board[i + 2][j - 1] not in pieces:
        moves.append([piece, i + 2, j - 1, row, column])

    if i + 2 < 8 and j + 1 < 8 and board[i + 2][j + 1] not in pieces:
        moves.append([piece, i + 2, j + 1, row, column])

    if i + 1 < 8 and j + 2 < 8 and board[i + 1][j + 2] not in pieces:
        moves.append([piece, i + 1, j + 2, row, column])

    if i - 1 >= 0 and j + 2 < 8 and board[i - 1][j + 2] not in pieces:
        moves.append([piece, i - 1, j + 2, row, column])

    if i - 2 >= 0 and j + 1 < 8 and board[i - 2][j + 1] not in pieces:
        moves.append([piece, i - 2, j + 1, row, column])

    if i - 2 >= 0 and j - 1 >= 0 and board[i - 2][j - 1] not in pieces:
        moves.append([piece, i - 2, j - 1, row, column])

    if i - 1 >= 0 and j - 2 >= 0 and board[i - 1][j - 2] not in pieces:
        moves.append([piece, i - 1, j - 2, row, column])

    allmoves = [i for i in moves if i[1] >= 0 and i[2] >= 0]
    allmoves.sort()
    # print("knight", allmoves)
    return allmoves


# Gets all possible bluejay moves
def bluejay_moves(player, position, board):
    row, column = position[0], position[1]
    i, j = row, column
    moves = []
    piece = 'B' if player == 'w' else 'b'

    pieces = player_pieces(player)
    take_out_pieces = player_take_out(player)

    while i - 1 >= 0 and j - 1 >= 0:
        if board[i - 1][j - 1] in take_out_pieces:
            moves.append([piece, i - 1, j - 1, row, column])
            break
        elif board[i - 1][j - 1] in pieces:
            break
        else:
            moves.append([piece, i - 1, j - 1, row, column])
        i = i - 1
        j = j - 1

    i, j = row, column
    while i + 1 < 8 and j - 1 >= 0:
        if board[i + 1][j - 1] in take_out_pieces:
            moves.append([piece, i + 1, j - 1, row, column])
            break
        elif board[i + 1][j - 1] in pieces:
            break
        else:
            moves.append([piece, i + 1, j - 1, row, column])
        i = i + 1
        j = j - 1

    i, j = row, column
    while i - 1 >= 0 and j + 1 < 8:
        if board[i - 1][j + 1] in take_out_pieces:
            moves.append([piece, i - 1, j + 1, row, column])
            break
        elif board[i - 1][j + 1] in pieces:
            break
        else:
            moves.append([piece, i - 1, j + 1, row, column])
        i = i - 1
        j = j + 1

    i, j = row, column
    while i + 1 < 8 and j + 1 < 8:
        if board[i + 1][j + 1] in take_out_pieces:
            moves.append([piece, i + 1, j + 1, row, column])
            break
        elif board[i + 1][j + 1] in pieces:
            break
        else:
            moves.append([piece, i + 1, j + 1, row, column])
        i = i + 1
        j = j + 1

    moves.sort()
    # print("bishop", moves)
    return moves


# Gets all possible kingfisher moves
def kingfisher_moves(player, position, board):
    row, column = position[0], position[1]
    i, j = row, column
    moves = []
    piece = 'K' if player == 'w' else 'k'

    pieces = player_pieces(player)
    take_out_pieces = player_take_out(player)

    if i - 1 >= 0:
        if board[i - 1][j] not in pieces:
            moves.append([piece, i - 1, j, row, column])
        if j - 1 >= 0 and board[i - 1][j - 1] not in pieces:
            moves.append([piece, i - 1, j - 1, row, column])
        if j + 1 < 8 and board[i - 1][j + 1] not in pieces:
            moves.append([piece, i - 1, j + 1, row, column])

    if i + 1 < 8:
        if board[i + 1][j] not in pieces:
            moves.append([piece, i + 1, j, row, column])
        if j - 1 >= 0 and board[i + 1][j - 1] not in pieces:
            moves.append([piece, i + 1, j - 1, row, column])
        if j + 1 < 8 and board[i + 1][j + 1] not in pieces:
            moves.append([piece, i + 1, j + 1, row, column])

    if j - 1 >= 0 and board[i][j - 1] not in pieces:
        moves.append([piece, i, j - 1, row, column])

    if j + 1 < 8 and board[i][j + 1] not in pieces:
        moves.append([piece, i, j + 1, row, column])

    moves = sorted(moves, key=lambda a: [a[1]])
    # print("king", moves)
    return moves


# Gets all possible Quetzal Moves
def quetzal_moves(player, position, board):
    diag_moves = bluejay_moves(player, position, board)
    strt_moves = robin_moves(player,
                             position, board)

    moves = diag_moves + strt_moves
    for move in moves:
        if player=='w':
            move[0] = 'Q'
        else:
            move[0]='q'

    moves.sort()
    # print("Queen", moves)
    return moves


# Gets all possible parakeet Moves
def parakeet_moves(player, position, board):
    row, column = position[0], position[1]
    i, j = row, column
    moves = []
    allmoves = list()
    piece = 'P' if player == 'w' else 'p'

    pieces = player_pieces(player)

    if row == 1 and board[row][column] == piece and board[row+1][column] not in pieces and board[row+2][column] not in pieces:
        moves.append([row + 1, column])
        moves.append([row + 2, column])
    elif board[i+1][j] not in own_piece:
        moves.append([i + 1, j])

    try:
        if board[i + 1][j + 1] != '.' and board[i + 1][j + 1] not in pieces:
            moves.append([i + 1, j + 1])
    except:
        pass
    try:
        if board[i + 1][j - 1] != '.' and board[i + 1][j - 1] not in pieces:
            moves.append([i + 1, j - 1])
    except:
        pass
    moves.sort()

    for move in moves:
        if player == "w":
            if move[0] == 7:
                allmoves += [['Q', move[0], move[1], row, column]]
            else:
                allmoves += [['P', move[0], move[1], row, column]]
        if player == "b":
            if move[0] == 0:
                allmoves += [['q', move[0], move[1], row, column]]
            else:
                allmoves += [['p', move[0], move[1], row, column]]

    return allmoves


# Removes invalid moves of a kind of piece
def rem_invalid_move(moves, player, board):
    curr_board = board[:]
    update_moves = moves
    invalid_moves = list()
    for move in update_moves:
        if player == 'w' and curr_board[move[1]][move[2]] in ('R', 'N', 'B', 'Q', 'K', 'P'):
            invalid_moves.append(move)
        elif player == 'b' and curr_board[move[1]][move[2]] in ('r', 'n', 'b', 'q', 'k', 'p'):
            invalid_moves.append(move)

    # have to add code to remove moves post taking the enemy piece
    for move in invalid_moves:
        update_moves.remove(move)

    # remove duplicates
    removed_duplicate = []
    for i in update_moves:
        if i not in removed_duplicate and i not in curr_board:
            removed_duplicate.append(i)

    return removed_duplicate


# Calls each kind of piece's get function and appends all possible moves of each piece
def successors(player, move):
    board = initial_board(move[0])
    moves = list()

    if player == "w":
        parakeet, robin, nighthawk, bluejay, quetzal, kingfisher = 'P', 'R', 'N', 'B', 'Q', 'K'
        curr_board = deepcopy(board)
    else:
        parakeet, robin, nighthawk, bluejay, quetzal, kingfisher = 'p', 'r', 'n', 'b', 'q', 'k'
        reversed_board = append(board)[::-1]
        curr_board = initial_board(reversed_board)

    robin_pos = curr_position(curr_board, robin)
    pawn_pos = curr_position(curr_board, parakeet)

    nighthawk_pos = curr_position(curr_board, nighthawk)
    bluejay_pos = curr_position(curr_board, bluejay)
    quetzal_pos = curr_position(curr_board, quetzal)
    king_pos = curr_position(curr_board, kingfisher)

    for rook in robin_pos:
        moves += robin_moves(player, rook, curr_board)

    for n in nighthawk_pos:
        moves += nighthawk_moves(player, n, curr_board)

    for b in bluejay_pos:
        moves += bluejay_moves(player, b, curr_board)

    for q in quetzal_pos:
        moves += quetzal_moves(player, q, curr_board)

    for k in king_pos:
        moves += kingfisher_moves(player, k, curr_board)

    for p in pawn_pos:
        moves += parakeet_moves(player, p, curr_board)

    rem_invalid_move( moves, player,curr_board)

    return possible_boards(curr_board, moves, player)


# Calculates and appends all possible moves and their evaluation values.
def possible_boards(board, moves, player):
    appended_all, all_moves = list(), list()
    for move in moves:
        curr_board = deepcopy(board)
        curr_board[move[1]][move[2]] = move[0]
        curr_board[move[3]][move[4]] = '.'
        if player == "w":
            appended_all += [append(curr_board)]
        else:
            appended_all += [append(curr_board)[::-1]]
    for move in appended_all:
        all_moves += [[move, evaluation(initial_board(move))]]

    # Max player descending order of scores(high to low), MIN play low to high, helps during alpha beta pruning
    if player == MAX:
        sorted(all_moves, key=lambda a: (a[1]), reverse=True)
    else:
        sorted(all_moves, key=lambda a: (a[1]))

    # Removing the initial board
    for each in all_moves:
        if each[0] == given_board:
            all_moves.remove(each)

    return all_moves


# for getting the proper evaluation values, we keep updating the number of pieces available and their cost
def taken_pieces(board):
    taken = {}
    if given_player == 'w':
        opp_player = 'b'
    else:
        opp_player = 'w'

    if given_player == 'w':
        opp_pieces = {'p': 8, 'b': 2, 'n': 2, 'r': 2, 'q': 1, 'k': 1}
        my_pieces = {'P': 8, 'B': 2, 'N': 2, 'R': 2, 'Q': 1, 'K': 1}
    else:
        my_pieces = {'p': 8, 'b': 2, 'n': 2, 'r': 2, 'q': 1, 'k': 1}
        opp_pieces = {'P': 8, 'B': 2, 'N': 2, 'R': 2, 'Q': 1, 'K': 1}
    taken[given_player] = my_pieces
    taken[opp_player] = opp_pieces

    for each in board.split()[0]:
        if each != '.':
            if each in my_pieces.keys():
                taken[given_player][each] = taken[given_player][each] - 1
            else:
                taken[opp_player][each] = taken[opp_player][each] - 1

    dic = {}
    dic[given_player] = {}
    dic[opp_player] = {}
    for player, items in taken.items():
        for piece, value in taken[player].items():
            if value != 0:
                if piece in my_pieces.keys():
                    dic[given_player][piece] = value
                else:
                    dic[opp_player][piece] = value

    return dic


# function which actually solves the chess board by calling the related functions.
def play_pichu(board, player, alphabeta=True):

    print ("Thinking! Please wait..." )
    possible_moves = successors(player, board)

    best_move = board
    # setting the initial score to least value so that we get a good move.
    best_score = -math.inf

    for move in possible_moves:
        if alphabeta:
            score = ab_mini_max(move, True, -math.inf, math.inf)
        else:
            score = mini_max(move, True, 1)

        if score > best_score:
            best_move = move
            best_score = score

    printable_board(initial_board(best_move[0]))
    print("\n")
    print("New board:")     
    print(best_move[0])
    exit()


# alpha beta pruning
def ab_mini_max(board, isMax, alpha, beta, current_depth=0):

    if current_depth == 1 and isMax:
        print ( 'depth 1 checking for  max' )

    if current_depth == MAX_DEPTH:
        return board[1]
    start = time.time()

    if isMax:
        while time.time() -  start < 20:
            best_value = -math.inf
            # Max player - set alpha
            for move in successors(MAX, board):

                max_move_score = ab_mini_max(move, False, alpha, beta, current_depth + 1)

                best_value = max([best_value, max_move_score])
                alpha = max(alpha, best_value)

                if (beta <= alpha):
                    break

            return best_value
    else:
        # Min player - set beta

        while time.time() - start < 20:
            best_value = math.inf


            for move in successors(MIN, board):

                min_move_score = ab_mini_max(move, True, alpha, beta, current_depth + 1)
                best_value = min(best_value, min_move_score)
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_value


# Mini max function which calculates the best moves
def mini_max(board, isMax, current_depth=0):
    if current_depth == MAX_DEPTH:
        return board[1]

    # is Max player
    if isMax:

        best_value = -math.inf
        for move in successors(MAX, board):
            max_move_score = mini_max(move, False, current_depth + 1)
            if max_move_score > best_value:
                best_value = max_move_score
            return best_value
    else:

        best_value = +math.inf
        for move in successors(MIN, board):
            min_move_score = mini_max(move, True, current_depth + 1)
            if min_move_score < best_value:
                best_value = min_move_score
            return best_value


# Calculates the evaluation of each piece
# https://chessprogramming.wikispaces.com/Simplified+evaluation+function
def calculate_eval_piece(board, mypiece, opp_piece, list):
    piece_pos_given_player = curr_position(board, mypiece)
    piece_pos_opposite_player = curr_position(board, opp_piece)
    if given_player == 'b':
        my_eval = 0.0
        for pos in piece_pos_given_player:
            my_eval += list[pos[0]][pos[1]]
        other_eval = 0.0
        for pos_o in piece_pos_opposite_player:
            other_eval += list[pos_o[0]][pos_o[1]]
    else:
        my_eval = 0.0
        for pos in piece_pos_given_player:
            x, y = pos[0] % 7, pos[1] % 7
            my_eval += list[x][y]
        other_eval = 0.0
        for pos_o in piece_pos_opposite_player:
            x, y = pos_o[0] % 7, pos_o[1] % 7
            other_eval += list[x][y]

    return (my_eval - other_eval)


# evaluates the value of each board for every given successor
def evaluation(board):
    board_position = {}
    board_position['p'] = [[0, 0, 0, 0, 0, 0, 0, 0],
                           [50, 50, 50, 50, 50, 50, 50, 50],
                           [10, 10, 20, 30, 30, 20, 10, 10],
                           [5, 5, 10, 25, 25, 10, 5, 5],
                           [0, 0, 0, 20, 20, 0, 0, 0],
                           [5, -5, -10, 0, 0, -10, -5, 5],
                           [5, 10, 10, -20, -20, 10, 10, 5],
                           [0, 0, 0, 0, 0, 0, 0, 0]
                           ]
    board_position['n'] = [[-50, -40, -30, -30, -30, -30, -40, -50],
                           [-40, -20, 0, 0, 0, 0, -20, -40],
                           [-30, 0, 10, 15, 15, 10, 0, -30],
                           [-30, 5, 15, 20, 20, 15, 5, -30],
                           [-30, 0, 15, 20, 20, 15, 0, -30],
                           [-30, 5, 10, 15, 15, 10, 5, -30],
                           [-40, -20, 0, 5, 5, 0, -20, -40],
                           [-50, -40, -30, -30, -30, -30, -40, -50]
                           ]
    board_position['b'] = [[-20, -10, -10, -10, -10, -10, -10, -20],
                           [-10, 0, 0, 0, 0, 0, 0, -10],
                           [-10, 0, 5, 10, 10, 5, 0, -10],
                           [-10, 5, 5, 10, 10, 5, 5, -10],
                           [-10, 0, 10, 10, 10, 10, 0, -10],
                           [-10, 10, 10, 10, 10, 10, 10, -10],
                           [-10, 5, 0, 0, 0, 0, 5, -10],
                           [-20, -10, -10, -10, -10, -10, -10, -20]]
    board_position['r'] = [[0, 0, 0, 0, 0, 0, 0, 0],
                           [5, 10, 10, 10, 10, 10, 10, 5],
                           [-5, 0, 0, 0, 0, 0, 0, -5],
                           [-5, 0, 0, 0, 0, 0, 0, -5],
                           [-5, 0, 0, 0, 0, 0, 0, -5],
                           [-5, 0, 0, 0, 0, 0, 0, -5],
                           [-5, 0, 0, 0, 0, 0, 0, -5],
                           [0, 0, 0, 5, 5, 0, 0, 0]]
    board_position['q'] = [[-20, -10, -10, -5, -5, -10, -10, -20],
                           [-10, 0, 0, 0, 0, 0, 0, -10],
                           [-10, 0, 5, 5, 5, 5, 0, -10],
                           [-5, 0, 5, 5, 5, 5, 0, -5],
                           [0, 0, 5, 5, 5, 5, 0, -5],
                           [-10, 5, 5, 5, 5, 5, 0, -10],
                           [-10, 0, 5, 0, 0, 0, 0, -10],
                           [-20, -10, -10, -5, -5, -10, -10, -20]]
    board_position['k'] = [[-30, -40, -40, -50, -50, -40, -40, -30],
                           [-30, -40, -40, -50, -50, -40, -40, -30],
                           [-30, -40, -40, -50, -50, -40, -40, -30],
                           [-30, -40, -40, -50, -50, -40, -40, -30],
                           [-20, -30, -30, -40, -40, -30, -30, -20],
                           [-10, -20, -20, -20, -20, -20, -20, -10],
                           [20, 20, 0, 0, 0, 0, 20, 20],
                           [20, 30, 10, 0, 0, 10, 30, 20]]

    if given_player == 'w':
        my_r_piece = 'R'
        my_p_piece = 'P'
        other_r_piece = 'r'
        other_p_piece = 'p'
        my_b_piece = 'B'
        other_b_piece = 'b'
        my_n_piece = 'N'
        other_n_piece = 'n'
        my_k_piece = 'K'
        other_k_piece = 'k'
        my_q_piece = 'Q'
        other_q_piece = 'q'

    else:
        my_r_piece = 'r'
        other_r_piece = 'R'
        my_p_piece = 'p'
        other_p_piece = 'P'
        my_b_piece = 'b'
        other_b_piece = 'B'
        my_n_piece = 'n'
        other_n_piece = 'N'
        my_k_piece = 'k'
        other_k_piece = 'K'
        my_q_piece = 'q'
        other_q_piece = 'Q'

    robin_diff = calculate_eval_piece(board, my_r_piece, other_r_piece, board_position['r'])
    parakeet_diff = calculate_eval_piece(board, my_p_piece, other_p_piece, board_position['p'])
    bluejay_diff = calculate_eval_piece(board, my_b_piece, other_b_piece, board_position['b'])
    q_diff = calculate_eval_piece(board, my_q_piece, other_q_piece, board_position['q'])
    k_diff = calculate_eval_piece(board, my_k_piece, other_k_piece, board_position['k'])
    n_diff = calculate_eval_piece(board, my_n_piece, other_n_piece, board_position['n'])

    k, q, r, b, n, p = 900, 90, 50, 30, 30, 10

    weighted_sum = k * k_diff + q * q_diff + r * robin_diff + p * parakeet_diff + n * n_diff + b * bluejay_diff

    return weighted_sum

#given_board = 'R.BQKBNRPPPPPPPP..N.................p........n..pppp.ppprnbqkb.r'
#given_player = 'w'
# python pichu.py w RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr 100
given_player = sys.argv[1]
given_board = sys.argv[2] 
given_time = sys.argv[3]


global parakeet, robin, nighthawk, bluejay, quetzal, kingfisher, own_piece, take_out
init_board = initial_board(given_board)

# abstraction to consider MAX and MIN players, different values for the kind of pieces
if given_player == "w":
    MAX, MIN = 'w', 'b'
    parakeet, robin, nighthawk, bluejay, quetzal, kingfisher = 'P', 'R', 'N', 'B', 'Q', 'K'
    own_piece = ('R', 'N', 'B', 'Q', 'K', 'P')
    take_out = ('r', 'n', 'b', 'q', 'k', 'p')
else:
    MAX, MIN = 'b', 'w'
    parakeet, robin, nighthawk, bluejay, quetzal, kingfisher = 'p', 'r', 'n', 'b', 'q', 'k'
    own_piece = ('r', 'n', 'b', 'q', 'k', 'p')
    take_out = ('R', 'N', 'B', 'Q', 'K', 'P')

printable_board(init_board)
print('-----')

# Start with Max player
def handler(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, handler)
signal.setitimer(signal.ITIMER_REAL, int(given_time))
try:
    play_pichu([given_board, evaluation(init_board)], MAX)
except:
    given_time = 0
finally:
    signal.alarm(0)
