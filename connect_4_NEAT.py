import neat
import pygame
import sys
import random
import numpy
import math
import pickle

WIDTH = 1100
HEIGHT =600
GAMETITLE = 'connect four'

DISC_RADIUS = 30

WHITE = (255,255,255)
GREY = (128, 128, 128)
RED = (255, 0, 0)
YELLOW = (255,255,0)
NIGHTBLUE = (34, 36, 64)
GREEN = (0, 255, 0)

PLAYER_COLOURS = [YELLOW, RED]


'''
Type of players:
1. Human
2. Neural Network
3. Random AI
4. Minimax AI
5. Training AI
'''
PLAYERS = ['Neural Network', 'Random AI', -1]

AI_depth = [1, 1]

Manual = False

Interval_time = 00

Max_games = 40


class Board(pygame.sprite.Sprite):
	def __init__(self, game):
		groups = game.all_sprites_group
		super().__init__(groups)
		self.image = pygame.Surface([WIDTH,HEIGHT])
		self.create_board()
		self.rect = self.image.get_rect()

	def create_board(self):
		x_gap = 100
		y_gap = HEIGHT//6
		self.coord_data = []
		self.board_data = numpy.zeros((7,6))
		self.sepatators = []
		for x in range(7):
			temp = []
			for y in range(6):
				pygame.draw.circle(self.image, GREY, (x*x_gap+50, y*y_gap+50), DISC_RADIUS)
				temp.append((x*x_gap+50-DISC_RADIUS, y*y_gap+50-DISC_RADIUS))
			pygame.draw.line(self.image, GREY, ((x)*x_gap, 0), ((x)*x_gap, HEIGHT), width=1)
			self.coord_data.append(temp)
			self.sepatators.append((x)*x_gap)
		pygame.draw.line(self.image, GREY, ((x+1)*x_gap, 0), ((x+1)*x_gap, HEIGHT), width=1)


class Discs(pygame.sprite.Sprite):
	def __init__(self,game, coord, colour):
		groups = game.discs_group, game.all_sprites_group
		super().__init__(groups)
		self.image = pygame.Surface([DISC_RADIUS*2,DISC_RADIUS*2])
		pygame.draw.circle(self.image,colour,(DISC_RADIUS,DISC_RADIUS), DISC_RADIUS)
		self.rect = self.image.get_rect()
		self.rect.x, self.rect.y = coord

class MinimaxClass():
	def __init__(self, game):
		self.game = game

	def Minimax(self, board, depth, maximizingPlayer, disc, opp_disc):
		valid_locations = self.game.available_moves()
		if depth == 0:
			return self.check_function(board, disc, opp_disc), self.count_up_scores(board, disc, opp_disc)
		if self.check_terminal(board, disc, opp_disc) == 1:
			return None, 100000000
		elif self.check_terminal(board, disc, opp_disc) == 2:
			return None, -100000000
		elif self.check_terminal(board, disc, opp_disc) == 3:
			return None, 0
		if maximizingPlayer:
			value = -math.inf
			best_x = random.choice(valid_locations)
			for x in valid_locations:
				temp_board = board.copy()
				temp_board[x][self.game.get_y_index(temp_board, x)] = disc
				score = self.Minimax(temp_board, depth-1, False, disc, opp_disc)[1]
				if score > value:
					value = score
					best_x = x
			return best_x, value
		else:
			value = math.inf
			best_x = random.choice(valid_locations)
			for x in valid_locations:
				temp_board = board.copy()
				temp_board[x][self.game.get_y_index(temp_board, x)] = opp_disc
				score = self.Minimax(temp_board, depth-1, True, disc, opp_disc)[1]
				if score < value:
					value = score
					best_x = x
			return best_x, value

	def check_function(self, board, disc, opp_disc):
		valid_locations = self.game.available_moves()
		best_score = -10000
		best_x = random.choice(valid_locations)
		for x in valid_locations:
			temp_board = board.copy()
			temp_board[x][self.game.get_y_index(temp_board, x)] = disc
			score = self.count_up_scores(temp_board, disc, opp_disc)
			if score > best_score:
				best_score = score
				best_x = x
		return best_x

	def check_terminal(self, board, disc, opp_disc):
		if self.game.win_detection(board, disc):
			return 1
		elif self.game.win_detection(board, opp_disc):
			return 2
		elif numpy.all(board) != 0:
			return 3
		else:
			return False

	def scoring(self, area, disc, opp_disc):
		score = 0
		if area.count(disc) == 4:
			score += 100
		elif area.count(disc) == 3 and area.count(0) == 1:
			score += 5
		elif area.count(disc) == 2 and area.count(0) == 2:
			score += 2
		if area.count(opp_disc) == 3 and area.count(0) == 1:
			score -= 8
		return score

	def count_up_scores(self, board, disc, opp_disc):
		score = 0
		data = board
		#center
		score += data[3].tolist().count(disc) * 3
		#verticals
		for x in data:
			for y in range(3):
				area = x[y:y+4].tolist()
				score += self.scoring(area, disc, opp_disc)
		#horizontals
		for y in data.T:
			for x in range(4):
				area = y[x:x+4].tolist()
				score += self.scoring(area, disc, opp_disc)
		#top right to bottom left diagonal
		for x in range(4):
			for y in range(3):
				area = [data[x+i][y+i] for i in range(4)]
				score += self.scoring(area, disc, opp_disc)
		#top left to bottom right diagonal
		for x in range(4,7):
			for y in range(3):
				area = [data[x-i][y+i] for i in range(4)]
				score += self.scoring(area, disc, opp_disc)
		return score



class Game():
	def __init__(self):
		pygame.init()
		size = (WIDTH, HEIGHT)
		self.screen = pygame.display.set_mode(size)
		pygame.display.set_caption(GAMETITLE)
		pygame.key.set_repeat(100, 50)
		self.mouse_pos = pygame.mouse.get_pos()
		self.font = pygame.font.SysFont('Comic Sans MS', 30)
		self.tracker = [0,0,0]
		self.minimax = MinimaxClass(self)

	def new_game(self, net, genome):
		self.all_sprites_group = pygame.sprite.Group()
		self.discs_group = pygame.sprite.Group()
		self.current_player = 1
		self.board = Board(self)
		self.last_time = pygame.time.get_ticks()
		self.net = net
		self.genome = genome

	def game_loop(self):
		self.run = True
		while self.run:
			self.events()
			self.update()
			self.draw()

	def wait_loop(self):
		self.wait = True
		while self.wait:
			self.wait_events()

	def events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.exit()
			if event.type == pygame.MOUSEBUTTONDOWN:
				mouse_presses = pygame.mouse.get_pressed()
				if mouse_presses[0]:
					self.mouse_pos = pygame.mouse.get_pos()
					self.create_disc()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_p:
					print(self.board.board_data)

	def wait_events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.exit()
			if event.type == pygame.MOUSEBUTTONDOWN:
				mouse_presses = pygame.mouse.get_pressed()
				if mouse_presses[0]:
					self.wait = False

	def create_disc(self):
		x_index = self.get_x_index()
		y_index = self.get_y_index(self.board.board_data, x_index)
		if y_index != -1 and x_index != -1:
			Discs(self, self.board.coord_data[x_index][y_index], PLAYER_COLOURS[self.current_player-1])
			self.board.board_data[x_index][y_index] = self.current_player
			if self.win_detection(self.board.board_data, self.current_player):
				self.game_over()
			elif PLAYERS[self.current_player-1] == 'Neural Network':
				self.genome.fitness += self.evaluate_move()
			if self.current_player == 1:
				self.current_player = 2
			else:
				self.current_player = 1

	def available_moves(self):
		temp = []
		for x in range(7):
			if self.board.board_data[x][0] == 0:
				temp.append(x)
		return temp

	def get_y_index(self, board, x_index):
		for y in range(5, -1, -1):
			if board[x_index][y] == 0:
				return y
		return -1

	def get_x_index(self):
		if numpy.all(self.board.board_data) != 0:
			self.current_player = 3
			self.game_over()
			return -1
		if PLAYERS[self.current_player-1] == 'Human':
			for i in range(6, -1, -1):
				if self.board.sepatators[i] < self.mouse_pos[0] < 700:
					print(self.minimax.Minimax(self.board.board_data, AI_depth[self.current_player-1], True, self.current_player, (1 if self.current_player == 2 else 2)))
					return i
		elif PLAYERS[self.current_player-1] == 'Neural Network':
			output = self.net.activate(self.board.board_data.flatten())
			index = output.index(max(output))
			while index not in self.available_moves():
				output[index] = -math.inf
				index = output.index(max(output))
			return index
		elif PLAYERS[self.current_player-1] == 'Minimax AI':
			return self.minimax.Minimax(self.board.board_data, AI_depth[self.current_player-1], True, self.current_player, (1 if self.current_player == 2 else 2))[0]
		elif PLAYERS[self.current_player-1] == 'Random AI':
			return random.choice(self.available_moves())
		elif PLAYERS[self.current_player-1] == 'Training AI':
			return self.training_AI()
		return -1

	def win_detection(self, data, player):
		for x in range(7):
			for y in range(3):
				if (data[x][y] == player) and (data[x][y+1] == player) and (data[x][y+2] == player) and (data[x][y+3] == player):
					if PLAYERS[self.current_player-1] == 'Neural Network':
						self.genome.fitness += 10
					return True
		for x in range(4):
			for y in range(6):
				if (data[x][y] == player) and (data[x+1][y] == player) and (data[x+2][y] == player) and (data[x+3][y] == player):
					if PLAYERS[self.current_player-1] == 'Neural Network':
						self.genome.fitness += 10
					return True
		for x in range(4):
			for y in range(3):
				if (data[x][y] == player) and (data[x+1][y+1] == player) and (data[x+2][y+2] == player) and (data[x+3][y+3] == player):
					if PLAYERS[self.current_player-1] == 'Neural Network':
						self.genome.fitness += 10
					return True
		for x in range(4):
			for y in range(3,6):
				if (data[x][y] == player) and (data[x+1][y-1] == player) and (data[x+2][y-2] == player) and (data[x+3][y-3] == player):
					if PLAYERS[self.current_player-1] == 'Neural Network':
						self.genome.fitness += 10
					return True

	def evaluate_move(self):
		return 0

	def training_AI(self):
		data = self.board.board_data
		opp_disc = (1 if self.current_player == 2 else 2)
		counter = 0
		for x in data:
			for y in range(3):
				area = x[y:y+4].tolist()
				if area.count(opp_disc) == 3 and area.count(0) == 1:
					return counter
			counter += 1
		return random.choice(self.available_moves())

	def game_over(self):
		self.run = False
		self.tracker[self.current_player-1] += 1

	def update(self):
		now = pygame.time.get_ticks()
		self.all_sprites_group.update()
		if PLAYERS[self.current_player-1] != 'Human' and now - self.last_time >= Interval_time and self.run and not Manual:
			self.create_disc()
			self.last_time = pygame.time.get_ticks()

	def draw(self):
		self.screen.fill(NIGHTBLUE)
		self.all_sprites_group.draw(self.screen)
		self.discs_group.draw(self.screen)
		if sum(self.tracker) == 0:
			winrate1 = 0
			winrate2 = 0
			drawrate = 0
		else:
			winrate1 = self.tracker[0]/sum(self.tracker)*100
			winrate2 = self.tracker[1]/sum(self.tracker)*100
			drawrate = self.tracker[2]/sum(self.tracker)*100
		texts = 'Player 1 : ' + PLAYERS[0]
		if PLAYERS[0] == 'Minimax AI':
			texts += ' d' + str(AI_depth[0])
		texts += '\nPlayer 2 : ' + PLAYERS[1]
		if PLAYERS[1] == 'Minimax AI':
			texts += ' d' + str(AI_depth[1])
		texts += '\n\nGames played : ' + str(sum(self.tracker))
		texts += '\nResults : ' + str(self.tracker)
		texts += '\nPlayer 1 win rate : ' + "{:.2f}".format(winrate1) + '%'
		texts += '\nPlayer 2 win rate : ' + "{:.2f}".format(winrate2) + '%'
		texts += '\nDraw rate : ' + "{:.2f}".format(drawrate) + '%'
		texts += '\ncurrent fitness : ' + str(self.genome.fitness)
		self.blit_texts(texts, WHITE, 710, 0, 40, self.font)
		pygame.display.flip()

	def blit_texts(self, texts, colour, x, y, y_intervals, font):
		textlist = texts.split('\n')
		counter = 0
		for line in textlist:
			if line != '':
				self.screen.blit(font.render(line, False, colour), (x, y + (y_intervals*counter)))
			counter += 1

	def exit(self):
		pygame.quit()
		sys.exit()

'''
game = Game()
while True:
	game.new_game()
	game.game_loop()
	if Manual or sum(game.tracker) >= Max_games:
		game.wait_loop()
'''

def eval_genomes(genomes, config):
	#print('Generation', p.generation+1)
	#counter = 0
	#total_fitness = 0
	#best_fitness = 0
	for genome_id, genome in genomes:
		#counter += 1
		game = Game()
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		genome.fitness = 0
		#print('Genome', counter)
		while sum(game.tracker) < Max_games:
			game.new_game(net, genome)
			game.game_loop()
		#print('result:', game.tracker, '\nfitness:', genome.fitness)
		#if genome.fitness > best_fitness:
		#	best_fitness = genome.fitness
		#total_fitness += genome.fitness
	#print('Generation end\nAverage fitness:', total_fitness/counter,'\nBest fitness:', best_fitness)



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))

winner = p.run(eval_genomes)

print('\nBest genome:\n{!s}'.format(winner))

with open('Trained_net', 'wb') as f:
	pickle.dump(winner, f)

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

Max_games = 10000
game = Game()
while True:
	game.new_game(winner_net, winner)
	game.game_loop()
	if Manual or sum(game.tracker) >= Max_games:
		game.wait_loop()