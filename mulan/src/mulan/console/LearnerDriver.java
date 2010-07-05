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

public class LearnerDriver {

	//private static final String LIST_LEARNERS_OPT = "l";
	private static final String HELP_OPT = "h";
	private static final String TRAIN_DATA_OPT = "trd";
	private static final String TEST_DATA_OPT = "tsd";
	private static final String LABELS_DEF_DATA_OPT = "ld";
	public static final String SEED_OPT = "s";
	public static final String SEED_OPT_LONG = "s";
	
	private static final Map<String, LearnerBuilder> learnerBuidersMap;
	
	static {
		// REMARK: new builders for learners should be added here...
		// To instantiate builders all the time all is not 'nice', 
		
		LearnerBuilder builder = null;
		
		learnerBuidersMap = new HashMap<String, LearnerBuilder>();
		builder = new BPMLLBuilder();
		learnerBuidersMap.put(builder.getSupportedType().getName(), builder);	
	}
	
	@SuppressWarnings("static-access")
	private Options getCommonOptions() {
		Options opt = new Options();
		
		opt.addOption(OptionBuilder.withLongOpt("train-data").hasArg().withType(File.class)
				.withDescription("Path to the data set which should be used for training a learner. The option is required.").create(TRAIN_DATA_OPT));
		opt.addOption(OptionBuilder.withLongOpt("test-data").hasArg().withType(File.class)
				.withDescription("Path to the data set which should be used for testing a learner. The option is required.").create(TEST_DATA_OPT));
		opt.addOption(OptionBuilder.withLongOpt("labels-definition").hasArg().withType(File.class)
				.withDescription("Path to the XML file with label definitions. The option is required.").create(LABELS_DEF_DATA_OPT));
		opt.addOption(OptionBuilder.withLongOpt("help").create(HELP_OPT));
		
		return opt;
	}
	
	public Set<String> getSupportedLearners(){
		return new HashSet<String>(learnerBuidersMap.keySet());
	}
	
	public void doLearningExperiment(String learnerName, String[] args){
		if(learnerName == null || learnerName.equals("")){
			throw new IllegalArgumentException("The learnerName must be set.");
		}
		
		LearnerBuilder builder = getBuilderForLearnerName(learnerName);
		if(builder == null){
			throw new IllegalArgumentException(String.format("The learner with name '%s' is not suppoerted.", learnerName));
		}
		
		Options options = getCommonOptions();	
		for(Object o : builder.getOptions().getOptions()){
			options.addOption((Option)o);
		}
		
		CommandLineParser parser = new GnuParser();
		try {
			CommandLine cmdLine = parser.parse(options, args);
			
			if(cmdLine.hasOption(HELP_OPT)){
	        	HelpFormatter formatter = new HelpFormatter();
	        	formatter.printHelp(learnerName, options, true);
			} else {
			
				cmdLine = parser.parse(options, args);
	        
				if(!cmdLine.hasOption(TRAIN_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", TRAIN_DATA_OPT));
				}
	        	File trainFile =(File)cmdLine.getParsedOptionValue(TRAIN_DATA_OPT);
	        	
	        	if(!cmdLine.hasOption(TEST_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", TEST_DATA_OPT));
				}
	        	File testFile = (File)cmdLine.getParsedOptionValue(TEST_DATA_OPT);
	        	
	        	if(!cmdLine.hasOption(LABELS_DEF_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", LABELS_DEF_DATA_OPT));
				}
	        	File labelsDefFile = (File)cmdLine.getParsedOptionValue(LABELS_DEF_DATA_OPT);
	        	
	    		if(!trainFile.exists()){
	    			throw new ParseException(String.format("The training file '%s' do not exists.", trainFile.getAbsoluteFile()));
	    		}
	    		if(!testFile.exists()){
	    			throw new ParseException(String.format("The testing file '%s' do not exists.", testFile.getAbsoluteFile()));
	    		}
	    		if(!labelsDefFile.exists()){
	    			throw new ParseException(String.format("The labels definition file '%s' do not exists.", labelsDefFile.getAbsoluteFile()));
	    		}
	    		
	        	MultiLabelLearner learner = builder.build(cmdLine);
	        	perfromLearning(learner, trainFile, testFile, labelsDefFile);
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
