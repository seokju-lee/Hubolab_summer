//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <random>
#include <functional>
#include <iostream>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();
    // heightMap_ = world_->addHeightMap("/home/seok-ju/raisim_ws/raisimLib/rsc/xmlScripts/heightMaps/zurichHeightMap.png", 0, 0, 500, 500, 0.005, -10);
    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal_c_arm/urdf/anymal_c_arm.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    ball_ = world_->addSphere(0.05, 0.005, "steel");

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_ + 1);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, -0.4, 0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, 0.0, 1.2, 0.0, 0.0, 1.3, 0.0;
    ball_->setPosition(-0.260684, 0.00105369, 1.22574);

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);

    jointPgain.setZero(); jointPgain.tail(6) << 150.0, 150.0, 150.0, 150.0, 150.0, 150.0;
    jointDgain.setZero(); jointDgain.tail(6) << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    jointPgain.segment(6, nJoints_-6).setConstant(50.0);
    jointDgain.segment(6, nJoints_-6).setConstant(0.2);
    
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 49;
    actionDim_ = nJoints_ + 1; actionMean_.setZero(nJoints_); actionStd_.setZero(nJoints_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_;
    actionStd_.setConstant(0.3);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    ball_->setPosition(-0.260684, 0.00105369, 1.22574);
    ball_->setExternalForce(0, {0, 0, -1.5});
    updateObservation();
  }

  float stepnew(const Eigen::Ref<EigenVec>& action, float& ranval) {
    /// action scaling
    pTarget12_ = action.head(18).cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(18) = pTarget12_;
    prob_ = 1/(1+exp(-action[18]));

    anymal_->setPdTarget(pTarget_, vTarget_);

    anymal_->getFramePosition("ee_link", ee_pos_);
    ball_pos_ = ball_->getPosition();
   
    std::vector<raisim::Vec<3UL>> ball_force;
    // std::random_device rd;
    // std::mt19937 gen(rd()); 
    // std::bernoulli_distribution d(prob_);
    // std::cout << "ranval: " << ranval << "\n";

    if (ranval < 0.7) {
      ball_->setExternalForce(0, {0, 0, 0});
    }
    else {
      ball_force = ball_->getExternalForce();
      if (sqrt(pow(ball_force[0], 2) + pow(ball_force[1], 2) + pow(ball_force[2], 2)) <= 0.1){
        ball_->setExternalForce(0, {0, 0, 0});
      }
      else {
        ball_->setExternalForce(0, {-15.0*(ball_pos_[0]-ee_pos_[0]), -15.0*(ball_pos_[1]-ee_pos_[1]), -15.0*(ball_pos_[2]-ee_pos_[2])});
      } 
    }

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    if (ball_Pos_[2] <= 0.05 and ball_Pos_[0] > 0){
      rewards_.record("ball_pos", sqrt(ball_Pos_[0] + pow(ball_Pos_[1], 2))/20);
    }
    
    rewards_.record("zeroVel", exp(-bodyLinearVel_.head(2).squaredNorm()));

    return rewards_.sum();
  }
 
  void updateObservation() {
    anymal_->getState(gc_, gv_);
    anymal_->getFramePosition("ee_link", ee_pos_);
    ball_Pos_ = ball_->getPosition();
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
    
    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(18), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(18), /// joint velocity
        ball_Pos_; /// ball position
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);
    updateObservation();
    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() or ball_Pos_[2] <= 0.05)
        return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  bool check_;
  float prob_;
  raisim::ArticulatedSystem* anymal_;
  raisim::Sphere* ball_;
  raisim::Box* box_;
  raisim::Vec<3> pose_, ee_initpos, ball_init_, ee_pos_, ball_init2_, ee_init2_;
  raisim::HeightMap* heightMap_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, actionMean2_, actionStd2_, obDouble_, rewards_set_, base_Vel, ranval_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, ball_Pos_, ball_pos_;
  Eigen::Vector4d box_rot_;
  std::set<size_t> footIndices_;
  float error_, error_1;
  std::random_device rd;
  
  float *rewards1_Address, *rewards2_Address, *rewardt_Address;
  float r1_sum_, r2_sum_, rt_sum_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}