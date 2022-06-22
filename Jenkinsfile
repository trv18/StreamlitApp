pipeline { // declare that you are writing a pipeline

    agent any // declare this build will run on any available Jenkins agent.
    
    stages { // where the work happens

        stage("build"){ // define stage name
            steps{
                echo 'building the application'    // script that executes command on jenkins server/agent
            }
        }

        stage("test"){ 
            steps{
                echo 'testing the application' 
            }
        }

        stage("deploy"){
            steps{
                echo 'deploying the application' 
            }
        }
    }   

}