/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Evaluation.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */

package mulan.evaluation;

/**
 * Simple aggregation class which provides all possible evaluation measure types.
 * The evaluation is providing measures for particular multi-label learner type.
 * Only measures applicable to evaluated learner will be provided. 
 * Measures which are not applicable will be null. The proper measures are set by 
 * {@link Evaluator} based on predefined rules.
 * 
 * @see Evaluator
 * 
 * @author Jozef Vilcek
 */
public class Evaluation {
	
	private LabelBasedMeasures labelBasedMeasures;
	private ExampleBasedMeasures exampleBasedMeasures;
	private RankingBasedMeasures rankingBasedMeasures;
    private ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures;
    private HierarchicalMeasures hierarchicalMeasures;
	
	public LabelBasedMeasures getLabelBasedMeasures() {
		return labelBasedMeasures;
	}

	protected void setLabelBasedMeasures(LabelBasedMeasures labelBasedMeasures) {
		this.labelBasedMeasures = labelBasedMeasures;
	}
	
	public ExampleBasedMeasures getExampleBasedMeasures() {
		return exampleBasedMeasures;
	}
	
	protected void setExampleBasedMeasures(ExampleBasedMeasures exampleBasedMeasures) {
		this.exampleBasedMeasures = exampleBasedMeasures;
	}

	public RankingBasedMeasures getRankingBasedMeasures() {
		return rankingBasedMeasures;
	}
	
	protected void setRankingBasedMeasures(RankingBasedMeasures rankingBasedMeasures) {
		this.rankingBasedMeasures = rankingBasedMeasures;
	}

	public HierarchicalMeasures getHierarchicalMeasures() {
		return hierarchicalMeasures;
	}

	protected void setHierarchicalMeasures(HierarchicalMeasures hierarchicalMeasures) {
		this.hierarchicalMeasures = hierarchicalMeasures;
	}

	protected void setConfidenceLabelBasedMeasures(ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures) {
		this.confidenceLabelBasedMeasures = confidenceLabelBasedMeasures;
	}

	public ConfidenceLabelBasedMeasures getConfidenceLabelBasedMeasures() {
		return confidenceLabelBasedMeasures;
	}

    @Override
	public String toString() {
		String description = "";

        if (exampleBasedMeasures != null) {
//		description += "Average predicted labels: " + this.numPredictedLabels + "\n";
            description += "========Example Based Measures========\n";
            description += "HammingLoss    : " + exampleBasedMeasures.getHammingLoss() + "\n";
            description += "Accuracy       : " + exampleBasedMeasures.getAccuracy() + "\n";
            description += "Precision      : " + exampleBasedMeasures.getPrecision() + "\n";
            description += "Recall         : " + exampleBasedMeasures.getRecall() + "\n";
            description += "Fmeasure       : " + exampleBasedMeasures.getFMeasure() + "\n";
            description += "SubsetAccuracy : " + exampleBasedMeasures.getSubsetAccuracy() + "\n";
        }
        if (labelBasedMeasures != null) {
            description += "========Label Based Measures========\n";
            description += "MICRO\n";
            description += "Precision    : " + labelBasedMeasures.getPrecision(Averaging.MICRO) + "\n";
            description += "Recall       : " + labelBasedMeasures.getRecall(Averaging.MICRO) + "\n";
            description += "F1           : " + labelBasedMeasures.getFMeasure(Averaging.MICRO) + "\n";
            description += "MACRO\n";
            description += "Precision    : " + labelBasedMeasures.getPrecision(Averaging.MACRO) + "\n";
            description += "Recall       : " + labelBasedMeasures.getRecall(Averaging.MACRO) + "\n";
            description += "F1           : " + labelBasedMeasures.getFMeasure(Averaging.MACRO) + "\n";
        }
        if (confidenceLabelBasedMeasures != null) {
            description += "MICRO\n";
            description += "AUC          : " + confidenceLabelBasedMeasures.getAUC(Averaging.MICRO) + "\n";
            description += "MACRO\n";
            description += "AUC          : " + confidenceLabelBasedMeasures.getAUC(Averaging.MACRO) + "\n";
        }
        if (rankingBasedMeasures != null) {
            description += "========Ranking Based Measures========\n";
            description += "One-error    : " + rankingBasedMeasures.getOneError() + "\n";
            description += "Coverage     : " + rankingBasedMeasures.getCoverage() + "\n";
            description += "Ranking Loss : " + rankingBasedMeasures.getRankingLoss() + "\n";
            description += "AvgPrecision : " + rankingBasedMeasures.getAvgPrecision() + "\n";
        }
        /*
        description += "========Per Class Measures========\n";
		for (int i = 0; i < numLabels(); i++) {
			description += "Label " + i + " Accuracy   :" + labelAccuracy[i] + "\n";
			description += "Label " + i + " Precision  :" + labelPrecision[i] + "\n";
			description += "Label " + i + " Recall     :" + labelRecall[i] + "\n";
			description += "Label " + i + " F1         :" + labelFmeasure[i] + "\n";
		}
		*/
		return description;
	}

    public String toCSV() {
		String description = "";

        if (exampleBasedMeasures != null) {
//		description += "Average predicted labels: " + this.numPredictedLabels + "\n";
            description += exampleBasedMeasures.getHammingLoss() + ";";
            description += exampleBasedMeasures.getAccuracy() + ";";
            description += exampleBasedMeasures.getPrecision() + ";";
            description += exampleBasedMeasures.getRecall() + ";";
            description += exampleBasedMeasures.getFMeasure() + ";";
            description += exampleBasedMeasures.getSubsetAccuracy() + ";";
        }
        if (labelBasedMeasures != null) {
            description += labelBasedMeasures.getPrecision(Averaging.MICRO) + ";";
            description += labelBasedMeasures.getRecall(Averaging.MICRO) + ";";
            description += labelBasedMeasures.getFMeasure(Averaging.MICRO) + ";";
            description += labelBasedMeasures.getPrecision(Averaging.MACRO) + ";";
            description += labelBasedMeasures.getRecall(Averaging.MACRO) + ";";
            description += labelBasedMeasures.getFMeasure(Averaging.MACRO) + ";";
        }
        if (confidenceLabelBasedMeasures != null) {
            description += confidenceLabelBasedMeasures.getAUC(Averaging.MICRO) + ";";
            description += confidenceLabelBasedMeasures.getAUC(Averaging.MACRO) + ";";
        }
        if (rankingBasedMeasures != null) {
            description += rankingBasedMeasures.getOneError() + ";";
            description += rankingBasedMeasures.getCoverage() + ";";
            description += rankingBasedMeasures.getRankingLoss() + ";";
            description += rankingBasedMeasures.getAvgPrecision() + ";";
        }
        /*
        description += "========Per Class Measures========\n";
		for (int i = 0; i < numLabels(); i++) {
			description += "Label " + i + " Accuracy   :" + labelAccuracy[i] + "\n";
			description += "Label " + i + " Precision  :" + labelPrecision[i] + "\n";
			description += "Label " + i + " Recall     :" + labelRecall[i] + "\n";
			description += "Label " + i + " F1         :" + labelFmeasure[i] + "\n";
		}
		*/
		return description;
	}
}
