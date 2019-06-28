require 'json'

module Assignment4
  class RandClassifier
  #   Trying to get the label randomly with the probablity for each.
  #   Threshold defined to provided minimum number to guess the perticular class in calse of n number of labels
    @probablity_threshold = nil

    def initialize
      @probablity_threshold = Hash.new{|h,k| h[k]=0.0}
    end

    def train dataset
      size = dataset.length
      probablity = Hash.new{|h,k| h[k]=0.0}

      dataset.each do |row|
        probablity[row["label"]] += 1.0/size
      end

      temp_variable = 0.0

      probablity.each do |key,value|
        @probablity_threshold[key] += value + temp_variable
        temp_variable += value
      end
    end

    def predict
      random_number = rand
      @probablity_threshold.each do |key, value|
        if rand < value
          return key
        end
      end
    end



  class NaiveBayesModel
    # Negative log likelihood
    def func dataset, w
      # class priro follows multinomial
      u = 0.0
      dataset.each {|r|
        s = r["features"]
        k = r["label"]
        u -= Math.log(w[k])
        s.keys.each {|j|
          p_jk = j.to_s + "_" + k.to_s
          u -= s[j] * Math.log(w[p_jk])
          }
        }
      return u
    end

    def grad dataset, w
      g = Hash.new {|h,k| h[k] = 0.0}

      dataset.each do |r|
        s = r["features"]
        k = r["label"]
        g[k] -= 1 / w[k]
        s.keys.each do |j|
          p_jk = j.to_s + "_" + k.to_s
          g[p_jk] -= s[j] / w[p_jk]
        end
      end

      return g
    end

    def predict row, w
      s = Hash.new {|h,k| h[k] = 0.0}

      ["0", "1"].each do |k|
        s[k] += Math.log(w[k])
        row["features"].keys.each do |j|
          p_jk = j.to_s + "_" + k.to_s
          s[k] += row["features"][j] * Math.log(w[p_jk])
        end
      end
      score= s.values.max
      return score
    end

    def adjust w
      w.each_key do |fname|
        w[fname] = [[0.001, w[fname]].max, 0.999].min
      end
    end
  end

end
