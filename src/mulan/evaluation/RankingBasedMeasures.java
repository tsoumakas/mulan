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
 *    RankingBasedMeasures.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */

package mulan.evaluation;

import java.util.ArrayList;
import mulan.classifier.MultiLabelOutput;
import mulan.core.Util;


public class RankingBasedMeasures {

	protected double oneError;
	protected double coverage;
	protected double rankingLoss;
	protected double avgPrecision;
	
	protected RankingBasedMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
		computeMeasures(output, trueLabels);
	}

    RankingBasedMeasures(RankingBasedMeasures[] arrayOfMeasures) {
		oneError = 0;
		coverage = 0;
		rankingLoss = 0;
		avgPrecision = 0;

        for (RankingBasedMeasures measures : arrayOfMeasures) {
            oneError += measures.getOneError();
            coverage += measures.getCoverage();
            rankingLoss += measures.getRankingLoss();
            avgPrecision += measures.getAvgPrecision();
        }

        int arraySize = arrayOfMeasures.length;
        oneError /= arraySize;
        coverage /= arraySize;
        rankingLoss /= arraySize;
        avgPrecision /= arraySize;
    }

    private void computeMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
		oneError = 0;
		coverage = 0;
		rankingLoss = 0;
		avgPrecision = 0;

        int numLabels = trueLabels[0].length;
        int examplesToIgnoreRankingLoss = 0;
        int examplesToIgnoreAvgPrecision = 0;
        int numInstances = output.length;
        for (int instanceIndex=0; instanceIndex<numInstances; instanceIndex++) {

            int[] ranks = output[instanceIndex].getRanking();

            //======one error related============
			int topRated;
            for (topRated=0; topRated<numLabels; topRated++)
                if (ranks[topRated] == 1)
                    break;
            if (!trueLabels[instanceIndex][topRated])
				oneError++;

			//======coverage related=============
			int howDeep = 0;
			for (int rank = numLabels; rank >= 1; rank--) {
                int indexOfRank;
                for (indexOfRank=0; indexOfRank<numLabels; indexOfRank++)
                    if (ranks[indexOfRank] == rank)
                        break;
				if (trueLabels[instanceIndex][indexOfRank]) {
					howDeep = rank-1;
					break;
				}
			}
			coverage += howDeep;

			// gather indexes of true and false labels
			// indexes of true and false labels
			ArrayList<Integer> trueIndexes = new ArrayList<Integer>();
			ArrayList<Integer> falseIndexes = new ArrayList<Integer>();
			for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
				if (trueLabels[instanceIndex][labelIndex]) {
					trueIndexes.add(labelIndex);
				} else {
					falseIndexes.add(labelIndex);
				}
			}

            //======ranking loss related=============
            if (trueIndexes.size() == 0 || falseIndexes.size() == 0)
                examplesToIgnoreRankingLoss++;
            else {
                int rolp = 0; // reversed ordered label pairs
                for (int k : trueIndexes)
                    for (int l : falseIndexes)
                        if (ranks[k] > ranks[l])
    //					if (output[instanceIndex].getConfidences()[trueIndexes.get(k)] <= output[instanceIndex].getConfidences()[falseIndexes.get(l)])
                            rolp++;
                rankingLoss += (double) rolp / (trueIndexes.size() * falseIndexes.size());
            }

            //======average precision related =============
            if (trueIndexes.size() == 0)
                examplesToIgnoreAvgPrecision++;
            else {
                double sum=0;
                for (int j : trueIndexes) {
                    double rankedAbove=0;
                    for (int k : trueIndexes)
                        if (ranks[k] <= ranks[j])
                            rankedAbove++;

                    sum += (rankedAbove / ranks[j]);
                }
                avgPrecision += (sum / trueIndexes.size());
            }
		}

        oneError /= numInstances;
		coverage /= numInstances;
		rankingLoss /= (numInstances - examplesToIgnoreRankingLoss);
		avgPrecision /= (numInstances - examplesToIgnoreAvgPrecision);
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
	
	/**
	 * Creates a summary string reporting values of all measures.
	 * @return the summary string
	 */
	public String toSummaryString(){
		String newLine = Util.getNewLineSeparator();
		StringBuilder summary = new StringBuilder();
		summary.append("========Ranking Based Measures========").append(newLine);
		summary.append("One-error\t: ").append(getOneError()).append(newLine);
		summary.append("Coverage\t: ").append(getCoverage()).append(newLine);
		summary.append("Ranking Loss\t: ").append(getRankingLoss()).append(newLine);
		summary.append("AvgPrecision\t: ").append(getAvgPrecision()).append(newLine);
		return summary.toString();
	}

}
