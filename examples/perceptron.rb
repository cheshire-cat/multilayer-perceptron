require_relative '../lib/multilayer_perceptron'

puts 'AND (1,1) perceptron:'

p = Neuron.new 2

puts '---not trained'
puts "1,1 - #{p.activate([1,1]).round}"
puts "1,0 - #{p.activate([1,0]).round}"
puts "0,1 - #{p.activate([0,1]).round}"
puts "0,0 - #{p.activate([0,0]).round}"

10000.times do
  p.train([1,1], 1)
  p.train([1,0], 0)
  p.train([0,1], 0)
  p.train([0,0], 0)
end

puts '---trained'
puts "1,1 - #{p.activate([1,1]).round}"
puts "1,0 - #{p.activate([1,0]).round}"
puts "0,1 - #{p.activate([0,1]).round}"
puts "0,0 - #{p.activate([0,0]).round}"
