# Inspired by dataset presented in course "End To End Machine Learning" by Brandon Rohrer https://github.com/brohrer

EXAMPLE_NUMBER = 100
EXAMPLE_LENGTH = 21
BLIP_LENGTH = 7

def blip_signal(symbol)
  case symbol
  when 'M'
    [1.0, 0.7, 0.4, 0.1, 0.4, 0.7, 1.0]
  when 'V'
    [-0.1, -0.4, -0.7, -1, -0.7, -0.4, -0.1]
  when 'N'
    [-0.7, 0.7, 0.4, 0.0, -0.4, -0.7, 0.7]
  when 'H'
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
  else
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  end
end

def random_blip_symbol
  ['M', 'V', 'N', 'H'].sample
end

puts "Generating blip dataset..."
outputFile = File.new('blips.csv', 'w')

(1..EXAMPLE_NUMBER).each do
  signal = Array.new(EXAMPLE_LENGTH, 0.0)

  blip_symbol = random_blip_symbol
  blip_location = rand(1..EXAMPLE_LENGTH - BLIP_LENGTH)
  blip = blip_signal(blip_symbol) 

  (0..BLIP_LENGTH - 1).each { |i| signal[blip_location + i] = blip[i] }

  outputFile.puts signal.unshift(blip_symbol).join(',')
end

outputFile.close
puts "Dataset generated."
