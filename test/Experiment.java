import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;
import mulan.classifier.AbstractMultiLabelClassifier;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.ExampleBasedEvaluation;
import mulan.evaluation.LabelBasedEvaluation;
import mulan.classifier.MultiLabelClassifier;

import weka.classifiers.Classifier;
import weka.core.FastVector;
import weka.core.Instances;

enum Measures
{
	/**
	 * Include columns with micro averaged evaluation measures.
	 */
	MICROLABEL,
	
	/**
	 * Include columns with macro averaged evaluation measures.
	 */	
	MACROLABEL,
	
	/**
	 * Include columns with example based evaluation measures.
	 */
	EXAMPLEBASED,
	
	/**
	 * Add columns with performance run time performance measures.
	 */
	TIMINGS
}

enum Evaluations
{
	/**
	 * Creates a single row of performance data per combination. 
	 * No arguments. Should be mutually exclusive with THRESHOLD evaluation.
	 */
	SIMPLE,
	
	/**
	 * Runs for a range of thresholds. 
	 * Takes 3 arguments, Double start, Double increment and Integer steps.
	 */
	THRESHOLD,
	
	/**
	 * Performs a crossvalidation. Takes 1 Integer argument, number of folds.
	 */
	CROSSVALIDATION,

	/**
	 * Include a column to identify each label and 4 extra measure columns for each individual 
	 * label: accuracy, recall, precision and fmeasure
	 */
	SPLITLABEL
}

/**
 * Encapsulate an Experiment setup and record all runs performed.
 * Supports object serialization.
 */

public class Experiment implements Serializable
{
	
//	public class ExperimentDriver
//	{
//		private Vector
//		public ExperimentDriver(Experiment exp)
//		{
//			
//		}
//		
//		public MultiLabelClassifier getClassifier()
//		{
//		
//		}
//	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 6088639125539260695L;

	
	/**
	 * Experiments should be named. The name is used as file 
	 * name if none is set.
	 */
	
	public String name = "Unnamed experiment";
	
	/**
	 * 
	 */
	public String description = "No description";
	
	/**
	 * The name of the file where this experiment will be serialized on calls to save().
	 */
	public String fileName;
	
	/**
	 * Total number of multiple labels.
	 */
	public int numLabels;

	/**
	 * Datasets and output are located relative to this directory.
	 */
	public String baseDir;
	
	
	/**
	 * If a single row of the experiment fails just mark it as failed and continue.
	 */
	public boolean breakOnException;

	/**
	 * Columns to include in the output.
	 */
	public EnumSet<Measures> measures;
	
	/**
	 * The classifiers to run the experiment on. A collection
	 * of fully qualified names of MultiLabelClassifiers mapped to
	 * a vector of options to pass. NOTE: Each vector element is a complete
	 * set of options to be parsed with weka.core.Utils.parseOptions.
	 */
	private HashMap<String, Vector<String>> classifiers;
	
	/**
	 * The underlying classifiers to combine with each 
	 * MultiLabelClassifier. A collection of fully qualified
	 * class names mapped to a vector of options to pass. 
	 * 
	 * NOTE: Each vector element is a complete set of options to be 
	 * parsed with weka.core.Utils.parseOptions.
   */
	private HashMap<String, Vector<String>> baseClassifiers;
	
	/**
	 * The datasets to iterate over.
	 */
	public Collection<DatasetReference> dataSets;
	
	/**
	 * The run history of this experiment.
	 */
	public List<Run> runs;
	
	
	public EnumMap<Evaluations, Object[]> evaluations;
	

	public Experiment()
	{
		dataSets = new ArrayList<DatasetReference>();
		baseClassifiers = new HashMap<String, Vector<String>>();
		classifiers     = new HashMap<String, Vector<String>>();
		measures =  EnumSet.noneOf(Measures.class);
		evaluations = new EnumMap<Evaluations, Object[]>(Evaluations.class);
		runs = new ArrayList<Run>();
	}
	
	/**
	 * Serialize this experiment to file.
	 * @param file
	 */
	
	public void save() throws Exception
	{
		String file = fileName;
		ObjectOutputStream stream = 
			new ObjectOutputStream(new FileOutputStream(file));
		stream.writeObject(this);
		stream.close();
	}
	
	public void saveAs(String file) throws Exception
	{
		fileName = file;
		save();
	}
	
	public static Experiment loadFromFile(String file) throws Exception
	{
		Experiment result = new Experiment();
		ObjectInputStream stream = 
			new ObjectInputStream(new FileInputStream(file));
		result = (Experiment) stream.readObject();
		stream.close();
		result.fileName = file;
		return result;
	}
	
	
	/**
	 * Shared method to avoid code duplication.
	 * @param destination
	 * @param classifier
	 * @param options
	 */
	private void appendClassifier(HashMap<String,Vector<String>> destination,
			String classifier, String options)
	{
		if (!destination.containsKey(classifier))
			destination.put(classifier, new Vector<String>());
		destination.get(classifier).add(options == null ? "" : options);
	}
	
	public void addClassifier(String classifier, String options)
	{
		appendClassifier(classifiers, classifier, options);
	}
	
	public void addBaseClassifier(String classifier, String options)
	{
		appendClassifier(baseClassifiers, classifier, options);
	}

	
	
	/**
	 * Some variables used during a run to keep track of columns in the output.
	 */
	private int numColumns;
	private HashMap<String, Integer> idxOfColumn; //used for lookup 
	private Vector<String> columnHeaders;
	private Vector<String> headerRow; //actual row used as first row of output
	
	/**
	 * Helper method for run(). Keep track of column headers.
	 * @param name
	 */
	private void appendColumn(String name)
	{
		columnHeaders.add(name);
		headerRow.add(name);
		idxOfColumn.put(name, numColumns++);
	}
	
	//Very ugly hack, extracted local loop variable to expose dataset reference.
	//Needed to set filenames in output. TODO: Redesign!
	private DatasetReference dsref;
	
	
	public Run run() throws Exception
	{ 
		//Pass an anonymous class implementing the interface ClassifierTweaker
		//to the real run() method.
		return run( new ClassifierTweaker()
	 	 {
			public boolean tweak(MultiLabelClassifier classifier, int step)
			{
				return false;
			}
	 	 }
		);
	}
	
	
	/**
	 * Run the experiment and record the results in the list
	 * of runs. 
	 */
	public Run run(ClassifierTweaker tweaker) throws Exception
	{
		Log.log("Begin experiment run [" + name + "]");
		Run run = new Run();
		
		//Ensure there is at least 1 base classifier.
		if (baseClassifiers.size() == 0) addBaseClassifier(null, "");
		
		initializeColumns();
		Vector<Vector<String>> rows = new Vector<Vector<String>>(0);
		rows.add(headerRow);

		boolean threshold     = evaluations.containsKey(Evaluations.THRESHOLD);
		boolean splitlabel    = evaluations.containsKey(Evaluations.SPLITLABEL);
		boolean crossValidate = evaluations.containsKey(Evaluations.CROSSVALIDATION);

		
		//Always iterate over the datasets first to avoid reloading from file
		for(DatasetReference dsref: dataSets)
		{
			this.dsref = dsref; //BAD
			dsref.baseDir = this.baseDir;


			Instances train = dsref.getTrain();
			
			//ensure references to test wont break
			Instances test  = crossValidate ?  
					new Instances("Unused", new FastVector(),0)
					: dsref.getTest();
					
			numLabels = dsref.numLabels;
			
			
			for(String strClassifier: classifiers.keySet())
				for(String strOptions: classifiers.get(strClassifier))
					for(String strBaseClassifier: baseClassifiers.keySet())
						for(String strBaseOptions: baseClassifiers.get(strBaseClassifier))
						{
							//Instantiate the classifiers
							String[] options     = weka.core.Utils.splitOptions(strOptions);
							String[] baseOptions = weka.core.Utils.splitOptions(strBaseOptions);
							
							AbstractMultiLabelClassifier classifier = (AbstractMultiLabelClassifier) Classifier.forName(strClassifier, options);
							if (strBaseClassifier != null)
							{
								Classifier baseClassifier = Classifier.forName(strBaseClassifier, baseOptions);
								classifier.setBaseClassifier(baseClassifier);
							}
							
							classifier.setNumLabels(dsref.numLabels);
							previous = System.nanoTime();
							Log.log(schemeOf((Classifier)classifier) + " : " + schemeOf(classifier.getBaseClassifier()));
						
							rows.addAll(evaluationStep(threshold,
										splitlabel, crossValidate, classifier,
										train, test));
						}
		}
		
		run.data = new String[rows.size()][numColumns];
		for(int r = 0; r < rows.size(); r++)
			for(int c = 0; c < numColumns; c++)
				run.data[r][c] = rows.get(r).get(c);
		
		//TODO: Add more info to the run
		runs.add(run);
		Log.log("End experiment run [" + name + "]");
		return run;
		
	}
	
	private void initializeColumns()
	{
		//Initialize the column headers
		idxOfColumn = new HashMap<String, Integer> ();
		columnHeaders = new Vector<String>();
		headerRow = new Vector<String>();
		numColumns = 0;
		appendColumn("Relation");	
		appendColumn("Datafile");
		appendColumn("Classifier scheme");
		appendColumn("Base Classifier scheme");
		appendColumn("Type");
		
		
		
		if (evaluations.containsKey(Evaluations.THRESHOLD)) appendColumn("Threshold");
		if (evaluations.containsKey(Evaluations.SPLITLABEL))
		{
			appendColumn("Label.Id");
			appendColumn("Label.Accuracy");
			appendColumn("Label.Recall");
			appendColumn("Label.Precision");
			appendColumn("Label.FMeasure");
		}
		
		for(Measures measure: measures)
		{
			if (measure == Measures.EXAMPLEBASED)
			{
				appendColumn("E.Hammingloss");     
				appendColumn("E.Accuracy");        
				appendColumn("E.Recall");          
				appendColumn("E.Precision");       
				appendColumn("E.SubsetAccuracy");  
				appendColumn("E.FMeasure");        
			}
			if (measure== Measures.MICROLABEL)
			{
				appendColumn("Micro.Accuracy");
				appendColumn("Micro.Recall");  
				appendColumn("Micro.Precision");
				appendColumn("Micro.FMeasure"); 
			}
			if (measure== Measures.MACROLABEL)
			{
				appendColumn("Macro.Accuracy");
				appendColumn("Macro.Recall");
				appendColumn("Macro.Precision");
				appendColumn("Macro.FMeasure");
			}			
			if (measure == Measures.TIMINGS)
			{
				appendColumn("Timing.Total ms");
				appendColumn("Timing.Training ms");
				appendColumn("Timing.Testing ms");
			}
		}		
	}
	
	private void setInitialColumns(Vector<String> row, String classifier, String baseClassifier, String dataset)
	{
		row.set(0, dataset);
		row.set(1, dsref.baseDir + dsref.trainFile + "|" + dsref.baseDir + dsref.testFile);
		row.set(2, classifier);
		row.set(3, baseClassifier);
		
	}
	

	protected Collection<Vector<String>> evaluationStep(boolean threshold, boolean splitlabel, boolean crossValidate, 
			AbstractMultiLabelClassifier classifier, 
			Instances train, Instances test)
	throws Exception
	{
		//Preallocate space 
		Vector<Vector<String>> result = new Vector<Vector<String>>(dsref.numLabels * 10);
		try
		{
			//TODO: Be careful! this will blow if threshold and crossvalidate are both set.
			if (!crossValidate)
				classifier.buildClassifier(train);
			
			Evaluation[] evals; 
			
			double start = 0, step = 0; //threshold params
			if (threshold) {
				//Extract arguments for thresholding
				Object[] args = evaluations.get(Evaluations.THRESHOLD);
				start = Double.parseDouble(args[0].toString());
				step = Double.parseDouble(args[1].toString());
				int steps = Integer.parseInt(args[2].toString());
				evals = new Evaluator().evaluateOverThreshold(classifier,
						test, start, step, steps);
			}
			else if (crossValidate)
			{
				//Array with single Evaluation
				evals = new Evaluation[3];
				InterlacedCV icv = new InterlacedCV(classifier, train, numLabels);
				
				int i = 0;
				for(AbstractMultiLabelClassifier.SubsetMappingMethod m : 
					new AbstractMultiLabelClassifier.SubsetMappingMethod[]{
						AbstractMultiLabelClassifier.SubsetMappingMethod.NONE, 
						AbstractMultiLabelClassifier.SubsetMappingMethod.GREEDY,
						AbstractMultiLabelClassifier.SubsetMappingMethod.PROBABILISTIC})
				{
					classifier.setSubsetMethod(m);
					evals[i++] = icv.evaluate();
				}
					
			}
			else
			{
				evals = new Evaluation[1];
				Evaluator evaluator = new Evaluator();
				if (crossValidate) {
					Object o = evaluations.get(Evaluations.CROSSVALIDATION)[0];
					evals[0] = o != null ? evaluator.crossValidate(
							classifier, train, (Integer) o) : evaluator
							.crossValidate(classifier, train);
				} else
					evals[0] = evaluator.evaluate(classifier, test);
			}
			int currentThresholdStep = 0;
			for (Evaluation evaluation : evals) {
				int numIterations = splitlabel ? dsref.numLabels : 1;
				for (int i = 0; i < numIterations; i++) {
					Vector<String> row = rowFromEvaluation(evaluation);
					setInitialColumns(row,
							schemeOf((Classifier) classifier),
							schemeOf(classifier.getBaseClassifier()), train
									.relationName()
									+ "/" + test.relationName());
					if (threshold) {
						setValue(row, "Type", "threshold");
						setValue(row, "Threshold", Double.toString(start
								+ step * currentThresholdStep));
					}
					if (splitlabel) {
						setValue(row, "Label.Id", Integer.toString(i));
						setValue(row, "Label.Accuracy", Double
								.toString(evaluation.getLabelBased()
										.accuracy(i)));
						setValue(row, "Label.Recall", Double
								.toString(evaluation.getLabelBased()
										.recall(i)));
						setValue(row, "Label.Precision", Double
								.toString(evaluation.getLabelBased()
										.precision(i)));
						setValue(row, "Label.FMeasure", Double
								.toString(evaluation.getLabelBased()
										.fmeasure(i)));
					}
					result.add(row);
				}
				currentThresholdStep++;
			}
			return result;
		}
		catch (Exception e)
		{
			if (breakOnException) throw e;
			Vector<Vector<String>> res = new  Vector<Vector<String>> ();
			res.add(errorRow(schemeOf((Classifier)classifier), schemeOf(classifier.getBaseClassifier()), train.relationName() + "/" + test.relationName(), e));
			return res;			
		}
	}
	
	private String join(String glue, String[] parts)
	{
		if (parts == null || parts.length==0) return "";
		StringBuilder builder = new StringBuilder(parts[0]);
		for(int i = 1; i < parts.length; i++)
			builder.append(" ").append(parts[i]);
		return builder.toString();
	}
	
	private String schemeOf(Classifier c)
	{
		return c.getClass().getName() + " " + join(" ", c.getOptions());
	}
	
	/**
	 * Convenience method to save a few keystrokes. 
	 * Called by rowFromEvaluation() below.
	 */
	private void setValue(Vector<String> row, String key, String value)
	{
		row.set(idxOfColumn.get(key), value);
	}
	
	long previous;
	
	protected Vector<String> rowFromEvaluation(Evaluation evaluation)
	{
		Vector<String> row = new Vector<String>(numColumns);
		row.setSize(numColumns);
		if (measures.contains(Measures.EXAMPLEBASED))
		{
			ExampleBasedEvaluation ebe = evaluation.getExampleBased();
			setValue(row,"E.Accuracy", Double.toString(ebe.accuracy()));
			setValue(row,"E.Hammingloss", Double.toString(ebe.hammingLoss()));
			setValue(row,"E.SubsetAccuracy", Double.toString(ebe.subsetAccuracy()));
			setValue(row,"E.Precision", Double.toString(ebe.precision()));
			setValue(row,"E.Recall", Double.toString(ebe.recall()));
			setValue(row,"E.FMeasure", Double.toString(ebe.fmeasure()));
		}

		LabelBasedEvaluation lbe = evaluation.getLabelBased();
		if (measures.contains(Measures.MACROLABEL))
		{
			lbe.setAveragingMethod(LabelBasedEvaluation.MACRO);
			
			setValue(row,"Macro.Accuracy", Double.toString(lbe.accuracy()));
			setValue(row,"Macro.Precision", Double.toString(lbe.precision()));
			setValue(row,"Macro.Recall", Double.toString(lbe.recall()));
			setValue(row,"Macro.FMeasure", Double.toString(lbe.fmeasure()));
		}

		if (measures.contains(Measures.MICROLABEL))
		{
			lbe.setAveragingMethod(LabelBasedEvaluation.MICRO);
			
			setValue(row,"Micro.Accuracy", Double.toString(lbe.accuracy()));
			setValue(row,"Micro.Precision", Double.toString(lbe.precision()));
			setValue(row,"Micro.Recall", Double.toString(lbe.recall()));
			setValue(row,"Micro.FMeasure", Double.toString(lbe.fmeasure()));
		}
		
		//TODO: Add timings
		if(measures.contains(Measures.TIMINGS))
		{
			long duration = System.nanoTime() - previous;
			previous = System.nanoTime();
			setValue(row, "Timing.Total ms", Double.toString(duration / 1000000));
			setValue(row, "Timing.Training ms", "not implemented");
			setValue(row, "Timing.Testing ms", "not implemented");
		}
		
		return row;
	}
	
	protected Vector<String> errorRow(String classifier, String base, String instances, Exception e)
	{
		Vector<String> result = new Vector<String>(numColumns);
		result.setSize(numColumns);
		result.set(0, instances);
		result.set(1, classifier);
		result.set(2, base);
		result.set(3, "**ERROR**");
		result.set(4, e.getMessage());
		return result;
	}

	public Run runTweaked()
	throws Exception
	{
		Log.log("Begin experiment run [" + name + "]");
		Run run = new Run();
		
		//Ensure there is at least 1 base classifier.
		if (baseClassifiers.size() == 0) addBaseClassifier(null, "");
		
		initializeColumns();
		Vector<Vector<String>> rows = new Vector<Vector<String>>(0);
		rows.add(headerRow);

		boolean threshold     = evaluations.containsKey(Evaluations.THRESHOLD);
		boolean splitlabel    = evaluations.containsKey(Evaluations.SPLITLABEL);
		boolean crossValidate = evaluations.containsKey(Evaluations.CROSSVALIDATION);

		
		//Always iterate over the datasets first to avoid reloading from file
		for(DatasetReference dsref: dataSets)
		{
			this.dsref = dsref; //BAD
			dsref.baseDir = this.baseDir;


			Instances train = dsref.getTrain();
			
			//ensure references to test wont break
			Instances test  = crossValidate ?  
					new Instances("Unused", new FastVector(),0)
					: dsref.getTest();
					
			numLabels = dsref.numLabels;
			
			
			for(String strClassifier: classifiers.keySet())
				for(String strOptions: classifiers.get(strClassifier))
					for(String strBaseClassifier: baseClassifiers.keySet())
						for(String strBaseOptions: baseClassifiers.get(strBaseClassifier))
						{
							//Instantiate the classifiers
							String[] options     = weka.core.Utils.splitOptions(strOptions);
							String[] baseOptions = weka.core.Utils.splitOptions(strBaseOptions);
							
							AbstractMultiLabelClassifier classifier = (AbstractMultiLabelClassifier) Classifier.forName(strClassifier, options);
							if (strBaseClassifier != null)
							{
								Classifier baseClassifier = Classifier.forName(strBaseClassifier, baseOptions);
								classifier.setBaseClassifier(baseClassifier);
							}
							
							classifier.setNumLabels(dsref.numLabels);
							previous = System.nanoTime();
							Log.log(schemeOf((Classifier)classifier) + " : " + schemeOf(classifier.getBaseClassifier()));
						
							rows.addAll(evaluationStep(threshold,
								splitlabel, crossValidate, classifier,
								train, test));
						}
		}
		
		run.data = new String[rows.size()][numColumns];
		for(int r = 0; r < rows.size(); r++)
			for(int c = 0; c < numColumns; c++)
				run.data[r][c] = rows.get(r).get(c);
		
		//TODO: Add more info to the run
		runs.add(run);
		Log.log("End experiment run [" + name + "]");
		return run;
		
	}	
}
