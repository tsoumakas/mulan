package mulan.evaluation;

import java.util.ArrayList;
import java.util.List;

import mulan.classifier.Ranking;


public class RankingMeasures {

	protected double oneError;
	protected double coverage;
	protected double rankingLoss;
	protected double avgPrecision;
	
	protected RankingMeasures(List<ModelEvaluationDataPair<Ranking>> rankingData){
		computeMeasures(rankingData);
	}

    private void computeMeasures(List<ModelEvaluationDataPair<Ranking>> predictions) {
		oneError = 0;
		coverage = 0;
		rankingLoss = 0;
		avgPrecision = 0;

        int numLabels = predictions.get(0).getNumLabels();

		for (ModelEvaluationDataPair<Ranking> pair : predictions) {
			// indexes of true and false labels
			ArrayList<Integer> true_indexes = new ArrayList<Integer>();
			ArrayList<Integer> false_indexes = new ArrayList<Integer>();


            List<Integer> ranks = pair.getModelOutput().getRanks();

            //======one error related============
			int topRated = ranks.indexOf(1);
            if (!pair.getTrueLabels().get(topRated))
				oneError++;

			//======coverage related=============
			int how_deep = 0;
			for (int j = numLabels; j >= 1; j--) {
				if (pair.getTrueLabels().get(ranks.indexOf(j))) {
					how_deep = j;
					break;
				}
			}
			coverage += how_deep;

			// gather indexes of true and false labels
			for (int j = 0; j < numLabels; j++) {
				if (pair.getTrueLabels().get(j)) {
					true_indexes.add(j);
				} else {
					false_indexes.add(j);
				}
			}

            //======ranking loss related=============
            int rolp = 0; // reversed ordered label pairs
			for (int k = 0; k < true_indexes.size(); k++) 
				for (int l = 0; l < false_indexes.size(); l++) 
					if (ranks.get(true_indexes.get(k)) > ranks.get(false_indexes.get(l)))
						rolp++;
			rankingLoss += (double) rolp / (true_indexes.size() * false_indexes.size());

			//======average precision related related=============
			double rel_rankj = 0;

			for (int j : true_indexes) {
				int jRank = ranks.get(j);
				int ranked_abovet = 0;

				// count the actually true above ranked labels
				for (int k = jRank-1; k >= 1; k--)
					if (pair.getTrueLabels().get(ranks.indexOf(k)))
						ranked_abovet++;
				rel_rankj += (double) (ranked_abovet + 1) / jRank; //+1 to include the current label
			}

			// divide by |Yi|
			rel_rankj /= true_indexes.size();

			avgPrecision += rel_rankj;
		}

        int numInstances = predictions.size();
        oneError /= numInstances;
		coverage /= numInstances;
		rankingLoss /= numInstances;
		avgPrecision /= numInstances;
    }

	
	
	public double getAvgPrecision(){
		return avgPrecision;
	}
	
	public double getRankingLoss(){
        return rankingLoss;
    }
	
	public double getCoverage(){
        return coverage;
    }
	
	public double getOneError(){
        return oneError;
    }
	
    @Override
	public String toString(){
        return "not implemented yet!";
	}

}
