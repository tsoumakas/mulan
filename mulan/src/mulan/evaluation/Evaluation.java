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

import mulan.core.Util;

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
    	String newLine = Util.getNewLineSeparator();
		StringBuilder summary = new StringBuilder();
        if (exampleBasedMeasures != null) {
        	summary.append(exampleBasedMeasures.toSummaryString()).append(newLine);
        }
        if (labelBasedMeasures != null) {
        	summary.append(labelBasedMeasures.toSummaryString()).append(newLine);
        }
        if (confidenceLabelBasedMeasures != null) {
        	summary.append(confidenceLabelBasedMeasures.toSummaryString()).append(newLine);
        }
        if (rankingBasedMeasures != null) {
        	summary.append(rankingBasedMeasures.toSummaryString()).append(newLine);
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
		return summary.toString();
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
