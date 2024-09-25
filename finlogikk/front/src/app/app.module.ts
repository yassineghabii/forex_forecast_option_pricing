import { APP_INITIALIZER, CUSTOM_ELEMENTS_SCHEMA, NgModule, isDevMode } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HeaderComponent } from './layout/header/header.component';
import { FooterComponent } from './layout/footer/footer.component';
import { ScrollComponent } from './components/scroll/scroll.component';
import { CarouselComponent } from './components/carousel/carousel.component';
import { AboutusComponent } from './components/aboutus/aboutus.component';
import { AboutUsBoxComponent } from './components/about-us-box/about-us-box.component';
import { ServicesComponent } from './components/services/services.component';
import { ServiceBoxComponent } from './components/service-box/service-box.component';
import { ChooesUsComponent } from './components/chooes-us/chooes-us.component';
import { ClientReviewComponent } from './components/client-review/client-review.component';
import { ServiceComponent } from './pages/service/service.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { TeamMemberComponent } from './components/team-member/team-member.component';
import { NgbCarouselModule, NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { ServiceWorkerModule } from '@angular/service-worker';
import { NgxSpinnerModule } from 'ngx-spinner';
import { HTTP_INTERCEPTORS, HttpClient, HttpClientModule } from '@angular/common/http';
import {FormsModule, ReactiveFormsModule} from "@angular/forms";

import { MatDialogModule } from '@angular/material/dialog';
import { ChooesUsBoxComponent } from './components/chooes-us-box/chooes-us-box.component';



import { MatIconModule } from '@angular/material/icon';
import {MatOptionModule} from "@angular/material/core";
import {MatSelectModule} from "@angular/material/select";
import {MatInputModule} from "@angular/material/input";
import {MatButtonModule} from "@angular/material/button";
import { ToastrModule } from 'ngx-toastr';

import { NgxPaginationModule } from 'ngx-pagination';

import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';

import {MatCardModule} from "@angular/material/card";
import { HttpTokenInterceptor } from './services/interceptor/http-token.interceptor';
import { KeycloakService } from './services/keycloak/keycloak.service';
import { OptionPricingComponent } from './components/option-pricing/option-pricing.component';
import {OptionPricingService} from "./services/option-pricing/option-pricing.service";
import { PrevisionComponent } from "./components/prevision/prevision.component";
import { PrevisionService } from "./services/prevision/prevision.service";
import { TopicComponent } from './components/topic/topic.component';
import { TopicDetailsComponent } from './components/topic-details/topic-details.component';
import { WindowRef } from "./services/services/window-ref.service";
import { StockDashboardComponent } from './components/stock-dashboard/stock-dashboard.component';

export function kcFactory(kcService: KeycloakService) {
  return () => kcService.init();

}

@NgModule({
  declarations: [
TeamMemberComponent,
    AppComponent,
    HeaderComponent,
    FooterComponent,
    ScrollComponent,
    CarouselComponent,
    AboutusComponent,
    AboutUsBoxComponent,
    ServicesComponent,
    ServiceBoxComponent,
    ChooesUsComponent,
    ChooesUsBoxComponent,

    ClientReviewComponent,


    ServiceComponent,
    DashboardComponent,
    TeamMemberComponent,
    OptionPricingComponent,
    PrevisionComponent,
    TopicComponent,
    TopicDetailsComponent,
    StockDashboardComponent,



  ],
    imports: [

        BrowserModule,
        BrowserAnimationsModule,
        MatIconModule,
        HttpClientModule,
        AppRoutingModule,
        NgbModule,
        MatDialogModule,
        NgbCarouselModule,

        NgxSpinnerModule,
        ReactiveFormsModule,
        FormsModule,
        MatIconModule,
        MatOptionModule,
        MatSelectModule,
        MatInputModule,
        MatButtonModule,
        MatCardModule,
        NgxPaginationModule,
        FontAwesomeModule,

    ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  providers: [
    WindowRef,
    HttpClient,
    {
      provide: HTTP_INTERCEPTORS,
      useClass: HttpTokenInterceptor,
      multi: true
    },
    {
      provide: APP_INITIALIZER,
      deps: [KeycloakService],
      useFactory: kcFactory,
      multi: true
    },
    OptionPricingService,
    PrevisionService

  ],
  bootstrap: [AppComponent],
})
export class AppModule {}
