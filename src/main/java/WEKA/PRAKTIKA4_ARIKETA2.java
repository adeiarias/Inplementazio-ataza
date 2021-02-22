package WEKA;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Date;
import java.util.Random;

public class PRAKTIKA4_ARIKETA2 {
    private static Instances test;
    private static PrintWriter printWriter;

    public static void main(String[] args) throws Exception {
        if(args.length != 3) {
            System.out.println("EZ DITUZU BEHAR DIREN PARAMETROAK SARTU");
            System.out.println("HURRENGO MODUAN JARRI BEHARKO DUZU:");
            System.out.println("java -jar IragarpenakEgin.jar  /paht/to/karpeta/NB.model  /paht/to/test_blind.arff  /paht/to/irteerako/karpeta/test_predictions.txt");
        }else{
            FileWriter fileWriter = new FileWriter(args[2]);
            printWriter = new PrintWriter(fileWriter);
            printWriter.println("EGIKARITUTAKO DATA -> " + new Date());
            printWriter.println("LEHENENGO ARGUMENTUA: NAIVEBAYES MODELOA KARGATZEKO HELBIDEA -> " + args[0]);
            printWriter.println("BIGARREN ARGUMENTUA: ARFF FITXATEGIAK -> " + args[1]);
            printWriter.println("HIRUGARREN ARGUMENTUA: ESTIMAZIOAK EGITEKO TXT -> " + args[2] + "\n");
            datuakKargatu(args[1]);
            ariketaEgin(args[0]);
            printWriter.close();
            fileWriter.close();
        }
    }

    private static void ariketaEgin(String path) throws Exception {
        Classifier naiveBayes = (Classifier)SerializationHelper.read(path);
        //OBJETUA GORDE
        SerializationHelper.write(path,naiveBayes);
        kalitateaEstimatu(naiveBayes);
    }

    private static void kalitateaEstimatu(Classifier naive) throws Exception {
        //CROSS VALIDATION
        Evaluation crossValidation = new Evaluation(data);
        crossValidation.crossValidateModel(naive,data,10,new Random(1));
        printWriter.println("CROSS VALIDATION --> ");
        printWriter.println("ACCURACY -> " + crossValidation.pctCorrect());
        printWriter.println("CONFUSION MATRIX -> " + crossValidation.toMatrixString() + "\n");

        //HOLD-OUT
        Randomize randomize = new Randomize();
        randomize.setRandomSeed(1);
        randomize.setInputFormat(data);
        Instances random = Filter.useFilter(data,randomize);

        RemovePercentage removePercentageTrain = new RemovePercentage();
        removePercentageTrain.setInputFormat(random);
        removePercentageTrain.setInvertSelection(true);
        removePercentageTrain.setPercentage(70.0);
        Instances train = Filter.useFilter(random,removePercentageTrain);

        RemovePercentage removePercentageTest = new RemovePercentage();
        removePercentageTest.setInputFormat(random);
        removePercentageTest.setInvertSelection(false);
        removePercentageTest.setPercentage(70.0);
        Instances test = Filter.useFilter(random,removePercentageTest);

        Evaluation holdOut = new Evaluation(train);
        holdOut.evaluateModel(naive,test);
        printWriter.println("HOLD-OUT --> ");
        printWriter.println("ACCURACY -> " + holdOut.pctCorrect());
        printWriter.println("CONFUSION MATRIX -> " + holdOut.toMatrixString());
    }

    private static void datuakKargatu(String arffFtixategia) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(arffFtixategia);
        test = source.getDataSet();
        if(test.classIndex() == -1) test.setClassIndex(test.numAttributes()-1);
    }

}
