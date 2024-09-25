import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import {OptionPricingComponent} from "./components/option-pricing/option-pricing.component";
import {PrevisionComponent} from "./components/prevision/prevision.component";
import {TopicComponent} from "./components/topic/topic.component";
import {TopicDetailsComponent} from "./components/topic-details/topic-details.component";
import { StockDashboardComponent } from "./components/stock-dashboard/stock-dashboard.component";

const routes: Routes = [
  { path: '', component: DashboardComponent},
  { path: 'topic/:id', component: TopicDetailsComponent},
  { path: 'option-pricing', component: OptionPricingComponent},
  { path: 'prevision', component: PrevisionComponent},
  { path: 'stock-dashboard', component: StockDashboardComponent},
  { path: 'topic', component: TopicComponent},
  { path: '**', redirectTo: '' },



];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
