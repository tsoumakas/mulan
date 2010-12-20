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

import mulan.classifier.MultiLabelLearner;
import mulan.core.MulanException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import weka.core.SerializationHelper;

/**
 * Drives the prediction experiment command line interface.
 * The driver performs:
 * - command line checks
 * - load learner state from specified file
 * - evaluate learner on test data set
 *  
 * @author Jozef Vilcek
 */
public class PredictionDriver {

	private static final String INPUT_FILE_OPT = "i";
	
	@SuppressWarnings("static-access")
	private Options getPredictionOptions() {
		Options opt = new Options();
		opt.addOption(OptionBuilder.withLongOpt("input").hasArg().withType(File.class)
				.withDescription("Path to the file containing stored learner state data. The option is required.").create(INPUT_FILE_OPT));
		opt.addOption(CommonOptions.getTestDataOption());
		opt.addOption(CommonOptions.getLabelsDefinitionOption());
		opt.addOption(CommonOptions.getHelpOption());
		
		return opt;
	}
	
	/**
	 * Performs prediction for specified input parameters
	 * 
	 * @param args input parameters
	 */
	public void doPredictionExperiment(String[] args){
		
		Options options = getPredictionOptions();	
		CommandLineParser parser = new GnuParser();
		try {
			CommandLine cmdLine = parser.parse(options, args);
			
			if(cmdLine.hasOption(CommonOptions.HELP_OPT)){
	        	HelpFormatter formatter = new HelpFormatter();
	        	formatter.printHelp("prediction", options, true);
			} else {
			
				cmdLine = parser.parse(options, args);
	        
				if(!cmdLine.hasOption(INPUT_FILE_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", INPUT_FILE_OPT));
				}
	        	File inputFile = (File)cmdLine.getParsedOptionValue(INPUT_FILE_OPT);
	        	
				if(!cmdLine.hasOption(CommonOptions.TEST_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", CommonOptions.TEST_DATA_OPT));
				}
	        	File testFile = (File)cmdLine.getParsedOptionValue(CommonOptions.TEST_DATA_OPT);
	        	
	        	if(!cmdLine.hasOption(CommonOptions.LABELS_DEF_DATA_OPT)){
					throw new ParseException(String.format("The option '%s' is required.", CommonOptions.LABELS_DEF_DATA_OPT));
				}
	        	File labelsDefFile = (File)cmdLine.getParsedOptionValue(CommonOptions.LABELS_DEF_DATA_OPT);
	        	
	        	if(!inputFile.exists()){
	    			throw new ParseException(String.format("The input file '%s' do not exists.", inputFile.getAbsoluteFile()));
	    		}
	    		if(!testFile.exists()){
	    			throw new ParseException(String.format("The testing file '%s' do not exists.", testFile.getAbsoluteFile()));
	    		}
	    		if(!labelsDefFile.exists()){
	    			throw new ParseException(String.format("The labels definition file '%s' do not exists.", labelsDefFile.getAbsoluteFile()));
	    		}
	    		
	    		// Successfully
	    		System.out.println("Loading the learner...");
	            MultiLabelLearner learner = null;
	            try{
	            	learner = (MultiLabelLearner) SerializationHelper.read(inputFile.getAbsolutePath());
	            	System.out.println(String.format("Learner '%s' loaded successfully.", learner.getClass().getName()));
	            }
	            catch(Exception ex){
	            	throw new MulanException("Failed to load learner state from the specified input file.", ex);
	            }
	            	
	            System.out.println("Performing prediction for test data...");

	            MultiLabelInstances testDataSet = new MultiLabelInstances(testFile.getAbsolutePath(), labelsDefFile.getAbsolutePath());
	            Evaluator evaluator = new Evaluator();
                Evaluation result = evaluator.evaluate(learner, testDataSet);
	            
                System.out.println();
        		System.out.println("Evaluatuion resutls:");
        		System.out.println(result.toString());
                
	        	System.out.println("Done...");
	        }
	        
	    }
	    catch(ParseException exp ) {
	        System.err.println("Commandline parsing failed. Reason: " + exp.getMessage());
	        System.out.println();
	        System.out.println("type: 'mulan predict --help' for help on usage");
	    }
	    catch(Exception e){
	    	System.err.println("Failed to perform prediction experiment:");
	    	e.printStackTrace(System.err);
	    }
	}
}
