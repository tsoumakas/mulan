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
 *    ICDM08EnsembleOfPrunedSets.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.experiments;

/**
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 */
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import weka.core.Utils;

public class PatternRecognition07MLkNN {

	public static void main(String[] args) {

		try {
			String path = Utils.getOption("path", args);
			String filestem = Utils.getOption("filestem", args);

			System.out.println("Loading the data set");
			MultiLabelInstances dataSet = new MultiLabelInstances(path
					+ filestem + ".arff", path + filestem + ".xml");

			Evaluator eval = new Evaluator();
			MultipleEvaluation results;
			List<Measure> measures = new ArrayList<Measure>(5);
			measures.add(new HammingLoss());
			measures.add(new OneError());
			measures.add(new Coverage());
			measures.add(new RankingLoss());
			measures.add(new AveragePrecision());

			int numFolds = 10;
			int numOfNeighbors;

			for (int i = 8; i <= 12; i++) {
				System.out.println("MLkNN Experiment for " + i + " neighbors:");
				numOfNeighbors = i;
				double smooth = 1.0;
				MLkNN mlknn = new MLkNN(numOfNeighbors, smooth);
				mlknn.setDontNormalize(true);
				// mlknn.setDebug(true);
				results = eval.crossValidate(mlknn, dataSet, measures, numFolds);
				System.out.println(results);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
