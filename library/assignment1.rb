require 'json'

module Assignment1
  def assignment1?
    true
  end


    def cross_validate x, folds, &block
      fold_size = x.size / folds
      subsets = []
      x.shuffle.each_slice(fold_size) do |subset|
        subsets << subset
      end
      i_folds = Array.new(folds) {|i| i}

      folds.times do |fold|
        test = subsets[fold]
        train = (i_folds - [fold]).flat_map {|t_fold| subsets[t_fold]}
        yield train, test, fold
      end
    end

    def mean_1 x
      sum = x.inject(0.0) {|u,v| u += v}
      sum / x.size
    end

    def stdev_1 x
      m = mean_1 x
      sum = x.inject(0.0) {|u,v| u += (v - m) ** 2.0}
      Math.sqrt(sum / (x.size - 1))
    end

    def confusion_matrix cls_names, x, predictions
      counts = Hash.new {|h,k| h[k] = Hash.new {|h,k| h[k] = 0}}
      cls_names.each {|i| cls_names.each{|j| counts[i][j] = 0}}

      x.each.with_index do |row, i|
        predicted_label = predictions[i].keys.first
        actual_label = row["label"]

        counts[predicted_label][actual_label] += 1
      end

      return counts
    end

    def accuracy conf_mat
      correct = 0.0
      sum = 0.0

      conf_mat.each_key do |pred|
        conf_mat[pred].each_key do |act|
          sum += conf_mat[pred][act]
          correct += conf_mat[pred][act] if pred == act
        end
      end

      correct / sum
    end
end
