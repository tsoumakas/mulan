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
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.console;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import mulan.classifier.MultiLabelLearner;
import mulan.console.builder.BPMLLBuilder;
import mulan.console.builder.LearnerBuilder;
import mulan.console.builder.MMPLearnerBuilder;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import weka.core.SerializationHelper;

/**
 * Drives the learning experiment command line interface. 
 * 
 * The driver performs:
 * - command line checks
 * - constructs correct learner
 * - train it on the specified train data set
 * - evaluate learner on test data set
 * - optionally stores learner state to the file for later reuse
 * 
 * @author Jozef Vilcek
 */
public class LearningDriver {

	private static final String OUTPUT_FILE_OPT = "o";
	private static final Map<String, LearnerBuilder> learnerBuidersMap;
	
	static {
		// REMARK: new builders for learners should be added here...
		// To instantiate builders all the time all is not 'nice', but also no issue for now 
		
		LearnerBuilder builder = null;
		learnerBuidersMap = new HashMap<String, LearnerBuilder>();
		
		builder = new BPMLLBuilder();
		learnerBuidersMap.put(builder.getSupportedType().getName(), builder);	
		builder = new MMPLearnerBuilder();
		learnerBuidersMap.put(builder.getSupportedType().getName(), builder);	
	}
	
	/**
	 * Returns set of learners supported by learning command line interface
	 * @return
	 */
	public Set<String> getSupportedLearners(){
		return new HashSet<String>(learnerBuidersMap.keySet());
	}
	
	/**
	 * Performs learning experiment for specified learner and parameters  
	 * 
	 * @param learnerName The name of a learner to be used
	 * @param args input parameters
	 */
	public void doLearningExperiment(String learnerName, String[] args){
		if(learnerName == null || learnerName.equals("")){
			throw new IllegalArgumentException("The learnerName must be set.");
		}
		
		LearnerBuilder builder = getBuilderForLearnerName(learnerName);
		if(builder == null){
			throw new IllegalArgumentException(String.format("The learner with name '%s' is not suppoerted.", learnerName));
		}
		
		Options options = getCommonLerningOptions();	
		for(Object o : builder.getOptions().getOptions()){
			options.addOption((Option)o);
		}
		
		CommandLineParser parser = new GnuParser();
		try {
			CommandLine cmdLine = parser.parse(options, args);
			
			if(cmdLine.hasOption(CommonOptions.HELP_OPT)){
	        	HelpFormatter formatter = new HelpFormatter();
	        	formatter.printHelp(learnerName, options, true);
			} else {
			
				cmdLine = parser.parse(options, args);
	        
				if(!cmdLine.hasOption(CommonOptions.TRAIN_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", CommonOptions.TRAIN_DATA_OPT));
				}
	        	File trainFile =(File)cmdLine.getParsedOptionValue(CommonOptions.TRAIN_DATA_OPT);
	        	
	        	if(!cmdLine.hasOption(CommonOptions.TEST_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", CommonOptions.TEST_DATA_OPT));
				}
	        	File testFile = (File)cmdLine.getParsedOptionValue(CommonOptions.TEST_DATA_OPT);
	        	
	        	if(!cmdLine.hasOption(CommonOptions.LABELS_DEF_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", CommonOptions.LABELS_DEF_DATA_OPT));
				}
	        	File labelsDefFile = (File)cmdLine.getParsedOptionValue(CommonOptions.LABELS_DEF_DATA_OPT);
	        	
	    		if(!trainFile.exists()){
	    			throw new ParseException(String.format("The training file '%s' do not exists.", trainFile.getAbsoluteFile()));
	    		}
	    		if(!testFile.exists()){
	    			throw new ParseException(String.format("The testing file '%s' do not exists.", testFile.getAbsoluteFile()));
	    		}
	    		if(!labelsDefFile.exists()){
	    			throw new ParseException(String.format("The labels definition file '%s' do not exists.", labelsDefFile.getAbsoluteFile()));
	    		}
	    		
	    		System.out.println("Creating learner instance ...");
	        	MultiLabelLearner learner = builder.build(cmdLine);
	        	perfromLearning(learner, trainFile, testFile, labelsDefFile);
	        	
	        	if(cmdLine.hasOption(OUTPUT_FILE_OPT)){
	        		System.out.println("Storing the model...");
	        		File outputFile = (File)cmdLine.getParsedOptionValue(OUTPUT_FILE_OPT);
	        		System.out.println(outputFile.getAbsolutePath());
	        		SerializationHelper.write(outputFile.getAbsolutePath(), learner);
	        	}
	        	System.out.println("Done...");
	        }
	        
	    }
	    catch(ParseException exp ) {
	        System.err.println("Commandline parsing failed. Reason: " + exp.getMessage());
	        System.out.println();
	        System.out.println("type: 'mulan learn  <learner_name> --help' for help on particular learner usage");
	    }
	    catch(Exception e){
	    	System.err.println("Failed to perform learning experiment:");
	    	e.printStackTrace(System.err);
	    }
	}

	@SuppressWarnings("static-access")
	private Options getCommonLerningOptions() {
		Options opt = new Options();
		
		opt.addOption(CommonOptions.getTrainDataOption());
		opt.addOption(CommonOptions.getTestDataOption());
		opt.addOption(CommonOptions.getLabelsDefinitionOption());
		opt.addOption(OptionBuilder.withLongOpt("output").hasArg().withType(File.class)
				.withDescription("Optional file path where to store trained learner state.").create(OUTPUT_FILE_OPT));
		opt.addOption(CommonOptions.getHelpOption());
		
		return opt;
	}
	
	private LearnerBuilder getBuilderForLearnerName(String learnerName){
		return learnerBuidersMap.containsKey(learnerName) ? 
				learnerBuidersMap.get(learnerName) : null;
	}
	
	private void perfromLearning(MultiLabelLearner learner, File trainFile, File testFile, File labelsDefFile) throws Exception{
		
		MultiLabelInstances trainDataSet = new MultiLabelInstances(trainFile.getAbsolutePath(), labelsDefFile.getAbsolutePath());
		MultiLabelInstances testDataSet = new MultiLabelInstances(testFile.getAbsolutePath(), labelsDefFile.getAbsolutePath());
		
		System.out.println("Training the learner ...");
		learner.build(trainDataSet);	
		
		Evaluator evaluator = new Evaluator();
		System.out.println("Evaluationg the learner on test data ...");
		Evaluation result = evaluator.evaluate(learner, testDataSet);
        
		System.out.println();
		System.out.println("Evaluatuion resutls:");
		System.out.println(result.toString());
	}
	

	
}
